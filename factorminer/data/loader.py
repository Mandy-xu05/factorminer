"""Market data loader supporting multiple formats and asset universes.

Loads OHLCV + amount data from CSV, Parquet, and HDF5 files. Supports
A-share universes (CSI500, CSI1000, HS300) and Binance crypto data.
Expected schema: datetime, asset_id, open, high, low, close, volume, amount.

The loader also accepts a small set of common aliases used by broker/data-vendor
exports, such as ``code``/``ticker`` for ``asset_id`` and ``amt`` for
``amount``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column ordering
REQUIRED_COLUMNS = [
    "datetime",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]

COLUMN_ALIASES = {
    "datetime": ["timestamp", "date", "time", "trade_date"],
    "asset_id": ["ticker", "symbol", "code", "stock_code", "ts_code", "instrument"],
    "open": ["open_price"],
    "high": ["high_price"],
    "low": ["low_price"],
    "close": ["close_price", "price"],
    "volume": ["vol"],
    "amount": ["amt", "turnover", "value", "traded_amount"],
}

# Well-known universe identifiers
UNIVERSE_ALIASES = {
    "csi500": "CSI500",
    "csi1000": "CSI1000",
    "hs300": "HS300",
    "binance": "Binance",
}

FileFormat = Literal["csv", "parquet", "hdf5"]


def read_panel_path(path: str | Path) -> pd.DataFrame:
    """Read a table file or a directory of Parquet row partitions."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Panel path not found: {path}")
    if not path.is_dir():
        return _read_file(path, _infer_format(path))
    files = sorted(path.rglob("*.parquet")) + sorted(path.rglob("*.pq"))
    if not files:
        raise ValueError(f"No Parquet files found in {path}")
    return pd.concat([_read_file(p, _infer_format(p)) for p in files], ignore_index=False)


def normalize_panel_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize date/code keys, including named index levels and YYYYMMDD dates."""
    df = df.copy()
    for key in ("datetime", "asset_id"):
        aliases = [key, *COLUMN_ALIASES[key]]
        if key not in df.columns:
            source = next((c for c in aliases if c in df.columns), None)
            if source is None:
                source = next((c for c in aliases if c in df.index.names), None)
                if source is not None:
                    df = df.reset_index(level=source)
            if source is None:
                raise ValueError(f"Panel missing {key!r} key")
            df = df.rename(columns={source: key})
    dates = df["datetime"]
    if pd.api.types.is_integer_dtype(dates):
        dates = dates.astype(str)
    df["datetime"] = pd.to_datetime(dates, errors="raise")
    df["asset_id"] = df["asset_id"].astype("string").str.strip()
    if df[["datetime", "asset_id"]].isna().any().any() or df["asset_id"].eq("").any():
        raise ValueError("Panel contains missing date/asset keys")
    return df


def read_market_directory(path: Path) -> pd.DataFrame:
    """Read per-field date/code/value files, or row-partitioned market tables."""
    fields = {"open": ["open"], "high": ["high"], "low": ["low"], "close": ["close"],
              "volume": ["volume", "vol"], "vwap": ["vwap"], "amount": ["amount", "amt"]}
    panels = []
    for field, aliases in fields.items():
        candidates = [path / f"{alias}{suffix}" for alias in aliases
                      for suffix in (".parquet", ".pq", "")]
        existing = [candidate for candidate in candidates if candidate.exists()]
        if len(existing) > 1:
            raise ValueError(f"Ambiguous sources for {field}: {existing}")
        if not existing:
            continue
        panel = normalize_panel_keys(read_panel_path(existing[0]))
        if "value" not in panel:
            raise ValueError(f"Per-field panel {existing[0]} requires a value column")
        panel = panel[["datetime", "asset_id", "value"]].rename(columns={"value": field})
        if panel.duplicated(["datetime", "asset_id"]).any():
            raise ValueError(f"Duplicate date/asset keys in {existing[0]}")
        panels.append(panel)
    if not panels:
        return normalize_panel_keys(read_panel_path(path))
    result = panels[0]
    for panel in panels[1:]:
        result = result.merge(panel, on=["datetime", "asset_id"], how="outer", validate="one_to_one")
    return result


def _infer_format(path: Path) -> FileFormat:
    suffix = path.suffix.lower()
    mapping = {
        ".csv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
    }
    fmt = mapping.get(suffix)
    if fmt is None:
        raise ValueError(f"Cannot infer format from extension '{suffix}'. "
                         f"Supported: {list(mapping.keys())}")
    return fmt  # type: ignore[return-value]


def _read_file(
    path: Path,
    fmt: FileFormat,
    hdf_key: str = "data",
) -> pd.DataFrame:
    """Read a single data file into a DataFrame."""
    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "hdf5":
        df = pd.read_hdf(path, key=hdf_key)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return df


def _validate_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Ensure required columns are present and normalise names."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    rename_map: dict[str, str] = {}
    missing: list[str] = []
    for req in REQUIRED_COLUMNS:
        if req in df.columns:
            continue
        candidates = [req, *COLUMN_ALIASES.get(req, [])]
        matched = None
        for candidate in candidates:
            original = cols_lower.get(candidate.lower().strip())
            if original is not None:
                matched = original
                break
        if matched is None:
            missing.append(req)
            continue
        rename_map[matched] = req
    # VWAP-only vendors need not invent turnover in unknown volume units.
    vwap_column = cols_lower.get("vwap")
    if vwap_column is not None:
        rename_map[vwap_column] = "vwap"
        if "amount" in missing:
            missing.remove("amount")
    if missing:
        raise ValueError(
            f"File {path} is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _coerce_types(
    df: pd.DataFrame,
    *,
    extra_numeric: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Ensure numeric types for OHLCV columns, extras, and datetime index."""
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["asset_id"] = df["asset_id"].astype(str)
    for col in OHLCV_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Preserve any additional numeric leaf columns (fundamentals, futures, ...)
    # rather than silently dropping them. Known non-numeric metadata is skipped.
    skip = {"datetime", "asset_id", "universe", "is_halt", *OHLCV_COLUMNS}
    candidates = list(extra_numeric or [])
    for col in df.columns:
        if col in skip or col in candidates:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
            continue
        # Attempt coercion for object columns that look numeric
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().any():
            df[col] = coerced
            candidates.append(col)
    for col in candidates:
        if col in df.columns and col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_market_data(
    path: str | Path,
    fmt: FileFormat | None = None,
    universe: str | None = None,
    asset_ids: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    hdf_key: str = "data",
) -> pd.DataFrame:
    """Load market data from a single file.

    Parameters
    ----------
    path : str or Path
        File path to the data source.
    fmt : str, optional
        File format (``"csv"``, ``"parquet"``, ``"hdf5"``). Inferred from
        the file extension when *None*.
    universe : str, optional
        Asset universe filter (e.g. ``"CSI500"``). Only assets belonging to
        the universe are kept. Requires an ``"universe"`` column in the data.
    asset_ids : sequence of str, optional
        Explicit list of asset identifiers to retain.
    start, end : str, optional
        ISO-formatted datetime strings for temporal filtering.
    hdf_key : str
        HDF5 dataset key (default ``"data"``).

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with columns from :data:`REQUIRED_COLUMNS` plus any
        extras present in the source file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.is_dir():
        df = read_market_directory(path)
    else:
        df = _read_file(path, fmt or _infer_format(path), hdf_key=hdf_key)
    logger.info("Loading %s from %s", "parquet directory" if path.is_dir() else fmt, path)
    df = _validate_columns(df, path)
    df = _coerce_types(df)

    # Universe filter
    if universe is not None:
        canon = UNIVERSE_ALIASES.get(universe.lower(), universe)
        if "universe" in df.columns:
            df = df[df["universe"] == canon]
            logger.info("Filtered to universe %s: %d rows", canon, len(df))
        else:
            logger.warning(
                "Universe filter '%s' requested but no 'universe' column found; "
                "filter skipped.",
                canon,
            )

    # Explicit asset filter
    if asset_ids is not None:
        asset_set = set(str(a) for a in asset_ids)
        df = df[df["asset_id"].isin(asset_set)]

    # Temporal filter
    if start is not None:
        df = df[df["datetime"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["datetime"] <= pd.Timestamp(end)]

    df = df.sort_values(["datetime", "asset_id"]).reset_index(drop=True)
    logger.info("Loaded %d rows, %d assets", len(df), df["asset_id"].nunique())
    return df


def load_multiple(
    paths: Sequence[str | Path],
    fmt: FileFormat | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load and concatenate market data from multiple files.

    All keyword arguments are forwarded to :func:`load_market_data`.
    """
    frames: list[pd.DataFrame] = []
    for p in paths:
        frames.append(load_market_data(p, fmt=fmt, **kwargs))
    if not frames:
        raise ValueError("No files provided to load_multiple")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["datetime", "asset_id"]).reset_index(drop=True)
    return df


def resample_market_data(
    df: pd.DataFrame,
    rule: str = "10min",
    *,
    closed: str = "left",
    label: str = "left",
    origin: str = "start_day",
) -> pd.DataFrame:
    """Resample canonical OHLCV market data per asset.

    Aggregation follows standard candle semantics:
    open = first, high = max, low = min, close = last, volume/amount = sum.
    ``vwap`` and ``returns`` are recomputed after aggregation.
    """
    if df.empty:
        return df.copy()

    work = _validate_columns(df.copy(), Path("<dataframe>"))
    work = _coerce_types(work)

    agg_spec: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum",
    }
    if "universe" in work.columns:
        agg_spec["universe"] = "first"

    frames: list[pd.DataFrame] = []
    for asset_id, group in work.groupby("asset_id", sort=True):
        group = group.sort_values("datetime").set_index("datetime")
        resampled = group.resample(
            rule,
            closed=closed,
            label=label,
            origin=origin,
        ).agg(agg_spec)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"], how="any")
        if resampled.empty:
            continue
        resampled["asset_id"] = asset_id
        frames.append(resampled.reset_index())

    if not frames:
        columns = ["datetime", "asset_id", *OHLCV_COLUMNS, "vwap", "returns"]
        return pd.DataFrame(columns=columns)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["datetime", "asset_id"]).reset_index(drop=True)
    out["vwap"] = np.divide(
        out["amount"],
        out["volume"],
        out=np.asarray(out["close"], dtype=np.float64).copy(),
        where=np.asarray(out["volume"], dtype=np.float64) > 1e-12,
    )
    out["returns"] = out.groupby("asset_id", sort=False)["close"].pct_change()

    ordered = ["datetime", "asset_id", *OHLCV_COLUMNS, "vwap", "returns"]
    extras = [col for col in out.columns if col not in ordered]
    return out[ordered + extras]


def to_numpy(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Convert a DataFrame to a numpy array of the specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Market data DataFrame.
    columns : sequence of str, optional
        Columns to include.  Defaults to :data:`OHLCV_COLUMNS`.

    Returns
    -------
    np.ndarray
        2-D float64 array of shape ``(n_rows, n_columns)``.
    """
    if columns is None:
        columns = OHLCV_COLUMNS
    return df[list(columns)].to_numpy(dtype=np.float64)
