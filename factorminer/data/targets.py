"""Read externally computed forward labels without modifying their timing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factorminer.data.loader import normalize_panel_keys, read_panel_path
from factorminer.data.tensor_builder import TargetSpec


def load_external_targets(
    market_keys: pd.DataFrame,
    data_config: Any,
    specs: list[TargetSpec],
) -> pd.DataFrame:
    """Left-align long-form labels by date/asset; never shift or fill missing labels.

    Values are decimal forward returns (0.05 means 5%), dated at signal time.
    ``targets`` still declares each label's horizon for train-boundary purging.
    """
    path = Path(data_config.target_path)
    source = read_panel_path(path)
    date_column = data_config.target_datetime_column
    asset_column = data_config.target_asset_column
    mapping = data_config.target_columns
    value_columns = list(dict.fromkeys(mapping[spec.name] for spec in specs))
    required = [date_column, asset_column, *value_columns]
    # Named parquet index levels are accepted as keys too.
    index_keys = [key for key in (date_column, asset_column)
                  if key not in source.columns and key in source.index.names]
    if index_keys:
        source = source.reset_index(level=index_keys)
    missing = set(required) - set(source.columns)
    if missing:
        raise ValueError(f"External targets missing columns: {sorted(missing)}")
    source = source[required].copy()
    source = source.rename(columns={date_column: "datetime", asset_column: "asset_id"})
    source = normalize_panel_keys(source)
    if source.duplicated(["datetime", "asset_id"]).any():
        raise ValueError("External targets contain duplicate (datetime, asset_id) keys")
    for column in value_columns:
        values = pd.to_numeric(source[column], errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan,
        )
        if np.isinf(values).any():
            raise ValueError(f"External target column {column!r} contains infinite values")
        source[column] = values
    keys = normalize_panel_keys(market_keys[["datetime", "asset_id"]])
    if data_config.target_mode == "daily_return":
        aligned = _compound_daily_returns(keys, source, mapping, specs)
    else:
        labels = source[["datetime", "asset_id"]].copy()
        for spec in specs:
            labels[spec.column_name] = source[mapping[spec.name]]
        aligned = keys.merge(labels, on=["datetime", "asset_id"], how="left", validate="one_to_one")
    for spec in specs:
        if not np.isfinite(aligned[spec.column_name].to_numpy(dtype=float)).any():
            raise ValueError(f"External target {spec.name!r} has no finite labels matching market keys")
    return aligned


def _compound_daily_returns(
    keys: pd.DataFrame,
    source: pd.DataFrame,
    mapping: dict[str, str],
    specs: list[TargetSpec],
) -> pd.DataFrame:
    """Daily row d is close[d]/close[d-1]-1; delay 1, horizon 5 uses d=t+2..t+6."""
    if keys.duplicated(["datetime", "asset_id"]).any():
        raise ValueError("Market panel contains duplicate date/asset keys")
    calendar = pd.Index(sorted(keys["datetime"].unique()))
    assets = pd.Index(sorted(keys["asset_id"].unique()))
    positions = calendar.get_indexer(keys["datetime"])
    codes = assets.get_indexer(keys["asset_id"])
    aligned = keys.copy()
    for spec in specs:
        panel = source.pivot(index="datetime", columns="asset_id", values=mapping[spec.name])
        daily = panel.reindex(index=calendar, columns=assets).to_numpy(dtype=float)
        if np.any(daily < -1):
            raise ValueError("Daily simple returns must be decimal values >= -1")
        end_offset = spec.entry_delay_bars + spec.holding_bars
        count = max(len(calendar) - end_offset, 0)
        compounded = np.full(daily.shape, np.nan, dtype=float)
        product = np.ones((count, len(assets)), dtype=float)
        for offset in range(spec.entry_delay_bars + 1, end_offset + 1):
            product *= 1.0 + daily[offset:offset + count]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            compounded[:count] = np.log(product) if spec.return_transform == "log" else product - 1.0
        compounded[~np.isfinite(compounded)] = np.nan
        aligned[spec.column_name] = compounded[positions, codes]
    return aligned
