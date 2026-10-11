"""Analysis-layer utilities for merger/lopsidedness workflows.

This module contains reusable, DataFrame-centric computation helpers.
It intentionally avoids file I/O, plotting, and UI dependencies.
"""

from __future__ import annotations

import ast
import copy
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu, norm


REQUIRED_COLUMNS = ("HaloID_Snap99", "SnapNum", "Mag_Gap", "Merger_Class")
TOP_BOTTOM_TOP_COL = "Most Lopsided (Top 1/3) Halo IDs (Snap 99)"
TOP_BOTTOM_BOTTOM_COL = "Least Lopsided (Bottom 1/3) Halo IDs (Snap 99)"
DEFAULT_METRIC_KEYS = [
    "Centroid_Offset",
    "MRL",
    "Pairwise",
    "Hemi_Contrast",
    "Normalised_RMS_Distance",
]
DEFAULT_MERGER_CLASSES = ["Stable", "Fly-by", "Infall", "Stripping"]


def _validate_input_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Validate common analysis input requirements.

    Required columns:
    - HaloID_Snap99
    - SnapNum
    - Mag_Gap
    - Merger_Class
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return {
        "n_rows": int(len(df)),
        "n_unique_halos": int(df["HaloID_Snap99"].nunique(dropna=True)),
        "snap_min": df["SnapNum"].min() if len(df) else None,
        "snap_max": df["SnapNum"].max() if len(df) else None,
    }


def _require_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    """Raise a descriptive error if a DataFrame is missing required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def parse_halo_id_list(id_str: Any) -> list[int]:
    """Parse comma-separated halo IDs from Top/Bottom table string fields."""
    if pd.isna(id_str) or id_str == "":
        return []

    parsed: list[int] = []
    for item in str(id_str).split(", "):
        if item:
            parsed.append(int(item))
    return parsed


def parse_top10_reference(df_top10: pd.DataFrame) -> dict[int, set[int]]:
    """Parse a top10 reference DataFrame into a snapshot->set(halo_id) map."""
    _require_columns(df_top10, ["SnapNum", "Halo_IDs"], "Top10 reference DataFrame")

    valid_top10_map: dict[int, set[int]] = {}
    for _, row in df_top10.iterrows():
        snap = int(row["SnapNum"])
        ids_raw = row["Halo_IDs"]

        if pd.isna(ids_raw):
            valid_top10_map[snap] = set()
            continue

        parsed_ids: list[int]
        if isinstance(ids_raw, (list, tuple, set, np.ndarray)):
            parsed_ids = [int(item) for item in ids_raw]
        else:
            parsed_obj: Any
            try:
                parsed_obj = ast.literal_eval(str(ids_raw))
            except (ValueError, SyntaxError):
                parsed_obj = []

            if isinstance(parsed_obj, (list, tuple, set, np.ndarray)):
                parsed_ids = [int(item) for item in parsed_obj]
            elif isinstance(parsed_obj, (int, float)) and not pd.isna(parsed_obj):
                parsed_ids = [int(parsed_obj)]
            else:
                parsed_ids = []

        valid_top10_map[snap] = set(parsed_ids)

    return valid_top10_map


def _extract_metric_value(row: pd.Series, key_cap: str, key_low: str) -> Any:
    """Extract a metric value from either canonical or lowercase alias column."""
    value = None
    if key_cap in row.index:
        value = row[key_cap]
    elif key_low in row.index:
        value = row[key_low]
    elif key_cap == "Centroid_Offset" and "Centroid_Distance" in row.index:
        value = row["Centroid_Distance"]

    if value is None or pd.isna(value):
        return None
        
    if key_cap == "Hemi_Contrast" and value is not None:
        value = abs(value)
        
    return value


def aggregate_satellite_metric_rows(
    df_satellite_rows: pd.DataFrame,
    *,
    is_top10: bool = False,
    valid_top10_map: dict[int, set[int]] | None = None,
    metrics_keys: list[str] | None = None,
    metrics_csv_keys: list[str] | None = None,
    reference_snapshot: int = 99,
) -> dict[int, dict[str, list[tuple[int, float]]]]:
    """Aggregate preloaded satellite-centric rows into raw overview structure."""
    _require_columns(df_satellite_rows, ["snap99_id", "SnapNum"], "Satellite metric rows")

    metric_keys = metrics_keys if metrics_keys is not None else list(DEFAULT_METRIC_KEYS)
    metric_csv_keys = (
        metrics_csv_keys
        if metrics_csv_keys is not None
        else ["centroid", "mrl", "pairwise", "hemi_contrast", "normalised_rms_distance"]
    )
    if len(metric_keys) != len(metric_csv_keys):
        raise ValueError("metrics_keys and metrics_csv_keys must have the same length.")

    valid_ids_snap99 = None
    if is_top10:
        valid_ids_snap99 = (valid_top10_map or {}).get(reference_snapshot, set())

    temp_data: dict[int, dict[str, list[tuple[int, float]]]] = {}
    for _, row in df_satellite_rows.iterrows():
        snap99_id = int(row["snap99_id"])
        if is_top10 and snap99_id not in valid_ids_snap99:
            continue

        snap = int(row["SnapNum"])
        if snap not in temp_data:
            temp_data[snap] = {key: [] for key in metric_keys}

        for key_cap, key_low in zip(metric_keys, metric_csv_keys):
            value = _extract_metric_value(row, key_cap, key_low)
            if value is not None:
                temp_data[snap][key_cap].append((snap99_id, value))

    return temp_data


def aggregate_host_metric_rows(
    df_host_rows: pd.DataFrame,
    lineage_df: pd.DataFrame,
    *,
    is_top10: bool = False,
    valid_top10_map: dict[int, set[int]] | None = None,
    metrics_keys: list[str] | None = None,
    metrics_csv_keys: list[str] | None = None,
) -> dict[int, dict[str, list[tuple[int, float]]]]:
    """Aggregate preloaded host-centric rows into raw overview structure."""
    _require_columns(df_host_rows, ["snapshot", "current_id"], "Host metric rows")

    metric_keys = metrics_keys if metrics_keys is not None else list(DEFAULT_METRIC_KEYS)
    metric_csv_keys = (
        metrics_csv_keys
        if metrics_csv_keys is not None
        else ["centroid", "mrl", "pairwise", "hemi_contrast", "normalised_rms_distance"]
    )
    if len(metric_keys) != len(metric_csv_keys):
        raise ValueError("metrics_keys and metrics_csv_keys must have the same length.")

    temp_data: dict[int, dict[str, list[tuple[int, float]]]] = {}
    for _, row in df_host_rows.iterrows():
        snap_num = int(row["snapshot"])
        current_id = int(row["current_id"])

        valid_ids_current = None
        if is_top10:
            valid_ids_current = (valid_top10_map or {}).get(snap_num, set())
            if current_id not in valid_ids_current:
                continue

        if snap_num not in temp_data:
            temp_data[snap_num] = {key: [] for key in metric_keys}

        lineage_col = f"HaloID_Snap_{snap_num}"
        if lineage_col not in lineage_df.columns:
            continue

        try:
            snap99_id = lineage_df[lineage_df[lineage_col] == current_id].index[0]
        except IndexError:
            continue

        for key_cap, key_low in zip(metric_keys, metric_csv_keys):
            value = _extract_metric_value(row, key_cap, key_low)
            if value is not None:
                temp_data[snap_num][key_cap].append((snap99_id, value))
            else:
                # Missing host-centric metric values are represented explicitly as NaN.
                temp_data[snap_num][key_cap].append((snap99_id, np.nan))

    return temp_data


def compute_percentile_band(
    values_by_snapshot: list[list[float]],
    lower_percentile: float = 16,
    upper_percentile: float = 84,
) -> dict[str, list[float]]:
    """Compute median and percentile bands per snapshot for aligned track values.

    Parameters
    ----------
    values_by_snapshot:
        A list where each item is a list of per-track values at one snapshot.
    lower_percentile, upper_percentile:
        Percentile bounds used for the spread band.
    """
    median: list[float] = []
    low_bound: list[float] = []
    high_bound: list[float] = []

    for values in values_by_snapshot:
        arr = np.array(values)
        if np.any(~np.isnan(arr)):
            with np.errstate(invalid="ignore"):
                median.append(float(np.nanmedian(arr)))
                low_bound.append(float(np.nanpercentile(arr, lower_percentile)))
                high_bound.append(float(np.nanpercentile(arr, upper_percentile)))
        else:
            median.append(float("nan"))
            low_bound.append(float("nan"))
            high_bound.append(float("nan"))

    return {"median": median, "low_bound": low_bound, "high_bound": high_bound}


def compute_percentile_band_from_matrix(
    matrix: np.ndarray,
    lower_percentile: float = 16,
    upper_percentile: float = 84,
    axis: int = 1,
) -> dict[str, np.ndarray]:
    """Compute median and percentile bands from a numeric matrix.

    This is used for pre-aligned snapshot-by-track matrices.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        median = np.nanmedian(matrix, axis=axis)
        low_bound = np.nanpercentile(matrix, lower_percentile, axis=axis)
        high_bound = np.nanpercentile(matrix, upper_percentile, axis=axis)

    return {"median": median, "low_bound": low_bound, "high_bound": high_bound}


def process_raw_overview_data(
    raw_data: dict[int, dict[str, list[tuple[int, float]]]],
    metrics_keys: list[str] | None = None,
    existing_system_table_data: dict[str, dict[str, dict[int, float]]] | None = None,
    latest_snapshot: int = 99,
    pairwise_cap: float = 90.0,
) -> dict[str, Any]:
    """Align raw metric observations and derive Snap99 Top/Bottom splits.

    Parameters
    ----------
    raw_data:
        Mapping of snapshot -> metric -> list of (halo_id_snap99, metric_value).
    metrics_keys:
        Metric keys to process. Defaults to canonical overview metrics.
    existing_system_table_data:
        Existing Top/Bottom mapping to preserve sequential merge behavior.
    latest_snapshot:
        Snapshot used for Top/Bottom split extraction (default: 99).
    pairwise_cap:
        Pairwise values above this cap are excluded from split logic.
    """
    metric_keys = list(metrics_keys) if metrics_keys is not None else list(DEFAULT_METRIC_KEYS)
    system_table_data = copy.deepcopy(existing_system_table_data) if existing_system_table_data else {}

    plot_data: dict[str, Any] = {"Tracked": {"SnapNum": [], **{k: [] for k in metric_keys}}}
    sorted_snaps = sorted(raw_data.keys())

    all_seen_ids: set[int] = set()
    for snap in sorted_snaps:
        snap_metrics = raw_data.get(snap, {})
        for metric in metric_keys:
            ids = [item[0] for item in snap_metrics.get(metric, [])]
            all_seen_ids.update(ids)

    master_sorted_ids = sorted(all_seen_ids)
    id_to_index = {halo_id: idx for idx, halo_id in enumerate(master_sorted_ids)}
    num_tracks = len(master_sorted_ids)
    filtered_snap_nums: list[int] = []

    for snap in sorted_snaps:
        snap_metrics = raw_data.get(snap, {})
        has_valid_data = any(bool(snap_metrics.get(metric)) for metric in metric_keys)
        if not has_valid_data:
            continue

        filtered_snap_nums.append(snap)

        for metric in metric_keys:
            raw_list = snap_metrics.get(metric, [])
            aligned_values = [np.nan] * num_tracks

            for halo_id, value in raw_list:
                if halo_id in id_to_index:
                    aligned_values[id_to_index[halo_id]] = value

            plot_data["Tracked"][metric].append(aligned_values)

            if snap == latest_snapshot and raw_list:
                unique_map: dict[int, float] = {item[0]: item[1] for item in raw_list}
                if metric == "Pairwise":
                    valid_items = [
                        (halo_id, value)
                        for halo_id, value in unique_map.items()
                        if value is not None and not np.isnan(value) and value <= pairwise_cap
                    ]
                    valid_items.sort(key=lambda item: item[1], reverse=True)
                else:
                    valid_items = [
                        (halo_id, value)
                        for halo_id, value in unique_map.items()
                        if value is not None and not np.isnan(value)
                    ]
                    valid_items.sort(key=lambda item: item[1])

                if valid_items:
                    n_total = len(valid_items)
                    n_select = int(n_total / 3)

                    if n_select > 0:
                        bottom_slice = valid_items[:n_select]
                        top_slice = valid_items[-n_select:]
                        top_ids = {halo_id: value for halo_id, value in top_slice}
                        bottom_ids = {halo_id: value for halo_id, value in bottom_slice}
                    else:
                        top_ids = {}
                        bottom_ids = {}

                    for halo_id in list(top_ids.keys()):
                        if halo_id in bottom_ids:
                            del bottom_ids[halo_id]

                    system_table_data[metric] = {
                        "Top 1/3": top_ids,
                        "Bottom 1/3": bottom_ids,
                    }

    plot_data["Tracked"]["SnapNum"] = filtered_snap_nums
    return {
        "plot_data": plot_data,
        "sorted_ids": master_sorted_ids,
        "system_table_data": system_table_data,
    }


def build_system_table_dataframe(
    system_table_data: dict[str, dict[str, dict[int, float]]],
    metrics_list: list[str] | None = None,
) -> pd.DataFrame:
    """Build display/export DataFrame for Top/Bottom Snap99 halo IDs."""
    metric_order = metrics_list if metrics_list is not None else list(DEFAULT_METRIC_KEYS)
    rows: list[dict[str, str]] = []

    for metric in metric_order:
        if metric not in system_table_data:
            continue

        top_ids = sorted(system_table_data[metric]["Top 1/3"].keys())
        bottom_ids = sorted(system_table_data[metric]["Bottom 1/3"].keys())
        rows.append(
            {
                "Metric": metric,
                TOP_BOTTOM_TOP_COL: ", ".join(map(str, top_ids)),
                TOP_BOTTOM_BOTTOM_COL: ", ".join(map(str, bottom_ids)),
            }
        )

    return pd.DataFrame(rows)


def compute_correlation_matrices(
    df_top_bottom: pd.DataFrame,
    df_merger: pd.DataFrame,
    merger_classes: list[str] | None = None,
) -> dict[str, Any]:
    """Compute Top/Bottom merger-class count and normalized matrices."""
    _require_columns(df_top_bottom, ["Metric", TOP_BOTTOM_TOP_COL, TOP_BOTTOM_BOTTOM_COL], "Top/Bottom DataFrame")
    _require_columns(df_merger, ["HaloID_Snap99", "Merger_Class"], "Merger DataFrame")

    class_order = merger_classes if merger_classes is not None else list(DEFAULT_MERGER_CLASSES)
    metrics = df_top_bottom["Metric"].unique().tolist()

    matrix_top = np.zeros((len(class_order), len(metrics)), dtype=int)
    matrix_bottom = np.zeros((len(class_order), len(metrics)), dtype=int)

    df_unique = df_merger.drop_duplicates(subset=["HaloID_Snap99"])
    class_map = pd.Series(df_unique["Merger_Class"].values, index=df_unique["HaloID_Snap99"]).to_dict()

    for col_idx, metric in enumerate(metrics):
        metric_row = df_top_bottom[df_top_bottom["Metric"] == metric].iloc[0]
        top_ids = parse_halo_id_list(metric_row[TOP_BOTTOM_TOP_COL])
        bottom_ids = parse_halo_id_list(metric_row[TOP_BOTTOM_BOTTOM_COL])

        for halo_id in top_ids:
            merger_class = class_map.get(halo_id, "Unknown")
            if merger_class in class_order:
                row_idx = class_order.index(merger_class)
                matrix_top[row_idx, col_idx] += 1

        for halo_id in bottom_ids:
            merger_class = class_map.get(halo_id, "Unknown")
            if merger_class in class_order:
                row_idx = class_order.index(merger_class)
                matrix_bottom[row_idx, col_idx] += 1

    matrix_total = matrix_top + matrix_bottom
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_top = matrix_top / matrix_total
        norm_bottom = matrix_bottom / matrix_total

    norm_top = np.nan_to_num(norm_top)
    norm_bottom = np.nan_to_num(norm_bottom)

    return {
        "results": {
            "metrics": metrics,
            "merger_classes": class_order,
            "matrix_top": matrix_top,
            "matrix_bottom": matrix_bottom,
            "norm_top": norm_top,
            "norm_bottom": norm_bottom,
        },
        "metadata": {
            "n_metrics": len(metrics),
            "n_merger_classes": len(class_order),
        },
    }


def build_contingency_inputs_from_top_bottom(
    df_top_bottom: pd.DataFrame,
    df_merger: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build per-metric contingency inputs with lopsided_flag and merger class."""
    _require_columns(df_top_bottom, ["Metric", TOP_BOTTOM_TOP_COL, TOP_BOTTOM_BOTTOM_COL], "Top/Bottom DataFrame")
    _require_columns(df_merger, ["HaloID_Snap99", "Merger_Class"], "Merger DataFrame")

    class_df = (
        df_merger.loc[:, ["HaloID_Snap99", "Merger_Class"]]
        .dropna(subset=["HaloID_Snap99", "Merger_Class"])
        .drop_duplicates(subset=["HaloID_Snap99"], keep="first")
        .copy()
    )

    metric_inputs: dict[str, pd.DataFrame] = {}
    for _, metric_row in df_top_bottom.iterrows():
        metric = metric_row["Metric"]
        top_ids = parse_halo_id_list(metric_row[TOP_BOTTOM_TOP_COL])
        bottom_ids = parse_halo_id_list(metric_row[TOP_BOTTOM_BOTTOM_COL])

        rows: list[dict[str, Any]] = []
        for halo_id in top_ids:
            rows.append({"HaloID_Snap99": halo_id, "lopsided_flag": "Top"})
        for halo_id in bottom_ids:
            rows.append({"HaloID_Snap99": halo_id, "lopsided_flag": "Bottom"})

        if not rows:
            metric_inputs[metric] = pd.DataFrame(columns=["HaloID_Snap99", "Merger_Class", "lopsided_flag"])
            continue

        flag_df = pd.DataFrame(rows).drop_duplicates(subset=["HaloID_Snap99", "lopsided_flag"])
        contingency_input = flag_df.merge(class_df, on="HaloID_Snap99", how="left")
        contingency_input = contingency_input.dropna(subset=["Merger_Class"])
        contingency_input = contingency_input.loc[:, ["HaloID_Snap99", "Merger_Class", "lopsided_flag"]]
        metric_inputs[metric] = contingency_input

    return metric_inputs


def compute_contingency_analysis(
    df: pd.DataFrame,
    reference_snapshot: int | None = None,
) -> dict[str, Any]:
    """Quantify association between lopsided grouping and merger class.

    Required columns:
    - HaloID_Snap99
    - Merger_Class
    - lopsided_flag (Top / Bottom)

    Notes:
    - Missing values are handled explicitly by dropping rows with missing required fields.
    - Only Top/Bottom lopsided flags are retained (case-insensitive normalization).
    - The function is pure: no I/O, no plotting, no mutation of input DataFrame.
    - reference_snapshot is accepted for API compatibility and is not used.
    - Returns overall association statistics plus cell-level standardized residuals
      and per-class Top/Bottom composition fractions.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required_cols = ["HaloID_Snap99", "Merger_Class", "lopsided_flag"]
    _require_columns(df, required_cols, "Contingency analysis input")

    work_df = df.loc[:, required_cols].copy()
    work_df = work_df.dropna(subset=required_cols)

    # Normalize lopsided_flag values to strict Top/Bottom buckets.
    normalized = work_df["lopsided_flag"].astype(str).str.strip().str.lower()
    work_df.loc[normalized.str.startswith("top"), "lopsided_flag"] = "Top"
    work_df.loc[normalized.str.startswith("bottom"), "lopsided_flag"] = "Bottom"

    work_df = work_df[work_df["lopsided_flag"].isin(["Top", "Bottom"])].copy()
    work_df = work_df.drop_duplicates(subset=["HaloID_Snap99", "Merger_Class", "lopsided_flag"])

    contingency_table = pd.crosstab(
        work_df["lopsided_flag"],
        work_df["Merger_Class"],
        dropna=False,
    )

    # Keep a stable row order when present.
    row_order = [row for row in ["Top", "Bottom"] if row in contingency_table.index]
    if row_order:
        contingency_table = contingency_table.reindex(row_order)

    # Per-class composition in Top vs Bottom groups.
    class_names = contingency_table.columns.tolist()
    top_counts = (
        contingency_table.loc["Top"].reindex(class_names, fill_value=0)
        if "Top" in contingency_table.index
        else pd.Series(0, index=class_names, dtype=float)
    )
    bottom_counts = (
        contingency_table.loc["Bottom"].reindex(class_names, fill_value=0)
        if "Bottom" in contingency_table.index
        else pd.Series(0, index=class_names, dtype=float)
    )

    total_top_count = float(top_counts.sum())
    total_bottom_count = float(bottom_counts.sum())

    if total_top_count > 0.0:
        top_fraction = top_counts.astype(float) / total_top_count
    else:
        top_fraction = pd.Series(np.nan, index=class_names, dtype=float)

    if total_bottom_count > 0.0:
        bottom_fraction = bottom_counts.astype(float) / total_bottom_count
    else:
        bottom_fraction = pd.Series(np.nan, index=class_names, dtype=float)

    class_fractions = pd.DataFrame(
        {
            "top_count": top_counts.astype(float),
            "bottom_count": bottom_counts.astype(float),
            "top_fraction": top_fraction.astype(float),
            "bottom_fraction": bottom_fraction.astype(float),
        }
    )
    class_fractions["fraction_diff"] = class_fractions["top_fraction"] - class_fractions["bottom_fraction"]
    class_fractions.index.name = "Merger_Class"

    standardized_residuals = pd.DataFrame(
        np.nan,
        index=contingency_table.index.copy(),
        columns=contingency_table.columns.copy(),
        dtype=float,
    )
    standardized_residuals.index.name = "lopsided_flag"

    residual_p_values = pd.DataFrame(
        np.nan,
        index=contingency_table.index.copy(),
        columns=contingency_table.columns.copy(),
        dtype=float,
    )
    residual_p_values.index.name = "lopsided_flag"

    if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return {
            "contingency_table": contingency_table,
            "chi2": float("nan"),
            "p_value": float("nan"),
            "cramers_v": float("nan"),
            "standardized_residuals": standardized_residuals,
            "residual_p_values": residual_p_values,
            "class_fractions": class_fractions,
        }

    chi2, p_value, _, expected = chi2_contingency(contingency_table.to_numpy())

    observed = contingency_table.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual_array = (observed - expected) / np.sqrt(expected)
    residual_array = np.where(expected > 0, residual_array, np.nan)
    standardized_residuals = pd.DataFrame(
        residual_array,
        index=contingency_table.index.copy(),
        columns=contingency_table.columns.copy(),
    )
    standardized_residuals.index.name = "lopsided_flag"

    p_val_array = 2 * norm.sf(np.abs(residual_array))
    p_val_array = np.where(expected > 0, p_val_array, np.nan)
    residual_p_values = pd.DataFrame(
        p_val_array,
        index=contingency_table.index.copy(),
        columns=contingency_table.columns.copy(),
    )
    residual_p_values.index.name = "lopsided_flag"

    n_obs = int(contingency_table.to_numpy().sum())
    n_rows, n_cols = contingency_table.shape
    min_dim = min(n_rows - 1, n_cols - 1)

    if n_obs == 0 or min_dim <= 0:
        cramers_v = float("nan")
    else:
        cramers_v = float(np.sqrt(chi2 / (n_obs * min_dim)))

    return {
        "contingency_table": contingency_table,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
        "standardized_residuals": standardized_residuals,
        "residual_p_values": residual_p_values,
        "class_fractions": class_fractions,
    }


def compute_merger_features(
    df: pd.DataFrame,
    reference_snapshot: int = 99,
    major_gap_threshold: float = 1.5,
    major_merger_classes: tuple[str, ...] = ("Infall",),
) -> pd.DataFrame:
    """Convert per-snapshot magnitude-gap history into halo-level features.

    Required columns:
    - HaloID_Snap99
    - SnapNum
    - Mag_Gap
    - Merger_Class

    Major-merger definition:
    - This definition is configurable via ``major_gap_threshold`` and
      ``major_merger_classes``.
    - A snapshot is flagged as a "major merger state" if either:
      1) ``Mag_Gap <= major_gap_threshold`` (default 1.5), OR
      2) ``Merger_Class`` is in ``major_merger_classes`` (default ``("Infall",)``),
         matched case-insensitively.
    - ``num_mergers`` is counted as the number of contiguous major-merger episodes
      across the snapshot sequence for each halo.

    Missing-data handling:
    - Rows with missing ``HaloID_Snap99`` or ``SnapNum`` are excluded.
    - ``Mag_Gap`` is coerced to numeric; non-numeric values become NaN.
    - Duplicate (HaloID_Snap99, SnapNum) rows are collapsed by first occurrence.
    - If no major-merger state is found for a halo, ``time_since_last_major_merger``
      is set to NaN.

    Returns
    -------
    pandas.DataFrame
        One row per ``HaloID_Snap99`` with:
        - HaloID_Snap99
        - Merger_Class
        - time_since_last_major_merger
        - num_mergers
        - mean_mag_gap
        - var_mag_gap
        - merger_time_fraction
    """
    _validate_input_dataframe(df)

    working_df = df.loc[:, ["HaloID_Snap99", "SnapNum", "Mag_Gap", "Merger_Class"]].copy()
    working_df["SnapNum"] = pd.to_numeric(working_df["SnapNum"], errors="coerce")
    working_df["Mag_Gap"] = pd.to_numeric(working_df["Mag_Gap"], errors="coerce")

    working_df = working_df.dropna(subset=["HaloID_Snap99", "SnapNum"]).copy()
    working_df = working_df.sort_values(["HaloID_Snap99", "SnapNum"])
    working_df = working_df.drop_duplicates(subset=["HaloID_Snap99", "SnapNum"], keep="first")

    major_class_norm = {label.strip().lower() for label in major_merger_classes}
    records: list[dict[str, Any]] = []

    for halo_id, group in working_df.groupby("HaloID_Snap99", sort=True):
        halo_df = group.sort_values("SnapNum").reset_index(drop=True)

        # Configurable major-merger state:
        #   Mag_Gap <= major_gap_threshold OR Merger_Class in major_merger_classes.
        class_norm = halo_df["Merger_Class"].astype(str).str.strip().str.lower()
        class_major_mask = class_norm.isin(major_class_norm)
        gap_major_mask = halo_df["Mag_Gap"].notna() & (halo_df["Mag_Gap"] <= major_gap_threshold)
        major_mask = class_major_mask | gap_major_mask

        major_episode_starts = major_mask & (~major_mask.shift(fill_value=False))
        num_mergers = int(major_episode_starts.sum())

        if bool(major_mask.any()):
            last_major_snapshot = float(halo_df.loc[major_mask, "SnapNum"].max())
            time_since_last_major_merger = float(reference_snapshot - last_major_snapshot)
        else:
            time_since_last_major_merger = float("nan")

        if halo_df["Mag_Gap"].notna().any():
            mean_mag_gap = float(halo_df["Mag_Gap"].mean())
            var_mag_gap = float(halo_df["Mag_Gap"].var(ddof=0))
        else:
            mean_mag_gap = float("nan")
            var_mag_gap = float("nan")

        n_observed = len(halo_df)
        merger_time_fraction = float(major_mask.sum() / n_observed) if n_observed > 0 else float("nan")

        merger_class_series = halo_df["Merger_Class"].dropna().astype(str).str.strip()
        if merger_class_series.empty:
            merger_class_value = pd.NA
        else:
            unique_classes = merger_class_series.unique()
            if len(unique_classes) == 1:
                merger_class_value = unique_classes[0]
            else:
                # Prefer the reference-snapshot class when present; otherwise use mode.
                ref_class_series = (
                    halo_df.loc[halo_df["SnapNum"] == reference_snapshot, "Merger_Class"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                if not ref_class_series.empty:
                    merger_class_value = ref_class_series.mode().iloc[0]
                else:
                    merger_class_value = merger_class_series.mode().iloc[0]

        records.append(
            {
                "HaloID_Snap99": halo_id,
                "Merger_Class": merger_class_value,
                "time_since_last_major_merger": time_since_last_major_merger,
                "num_mergers": num_mergers,
                "mean_mag_gap": mean_mag_gap,
                "var_mag_gap": var_mag_gap,
                "merger_time_fraction": merger_time_fraction,
            }
        )

    features_df = pd.DataFrame.from_records(records)
    if not features_df.empty:
        features_df = features_df.sort_values("HaloID_Snap99").reset_index(drop=True)

    return features_df


def compare_distributions(
    df: pd.DataFrame,
    reference_snapshot: int | None = None,
    feature_column: str | None = None,
) -> dict[str, Any]:
    """Compare Top vs Bottom lopsided groups for a selected numeric feature.

    Required columns:
    - lopsided_flag
    - at least one numeric feature column, or explicit ``feature_column`` input

    Notes:
    - Uses two-sided Kolmogorov-Smirnov and Mann-Whitney U tests.
    - Missing values are handled explicitly by dropping rows where
      ``lopsided_flag`` or the selected feature is missing/non-numeric.
    - Lopsided labels are normalized case-insensitively to ``Top``/``Bottom``.
    - reference_snapshot is accepted for API compatibility and is not used.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    _require_columns(df, ["lopsided_flag"], "Distribution comparison input")

    if feature_column is None:
        candidate_columns = [col for col in df.columns if col != "lopsided_flag"]
        numeric_candidates = [
            col for col in candidate_columns if pd.api.types.is_numeric_dtype(df[col])
        ]

        if len(numeric_candidates) == 1:
            feature_column = numeric_candidates[0]
        elif len(numeric_candidates) == 0:
            raise ValueError(
                "No numeric feature column found. Provide a DataFrame with at least one numeric feature "
                "or pass feature_column explicitly."
            )
        else:
            raise ValueError(
                "Multiple numeric feature columns found. Pass feature_column explicitly."
            )
    else:
        _require_columns(df, [feature_column], "Distribution comparison input")

    work_df = df.loc[:, ["lopsided_flag", feature_column]].copy()
    work_df[feature_column] = pd.to_numeric(work_df[feature_column], errors="coerce")

    normalized_flag = work_df["lopsided_flag"].astype(str).str.strip().str.lower()
    work_df.loc[normalized_flag.str.startswith("top"), "lopsided_flag"] = "Top"
    work_df.loc[normalized_flag.str.startswith("bottom"), "lopsided_flag"] = "Bottom"

    work_df = work_df[work_df["lopsided_flag"].isin(["Top", "Bottom"])].copy()
    work_df = work_df.dropna(subset=["lopsided_flag", feature_column])

    top_values = work_df.loc[work_df["lopsided_flag"] == "Top", feature_column].to_numpy(dtype=float)
    bottom_values = work_df.loc[work_df["lopsided_flag"] == "Bottom", feature_column].to_numpy(dtype=float)

    if len(top_values) == 0 or len(bottom_values) == 0:
        return {
            "ks_stat": float("nan"),
            "ks_p_value": float("nan"),
            "mw_stat": float("nan"),
            "mw_p_value": float("nan"),
        }

    ks_result = ks_2samp(top_values, bottom_values, alternative="two-sided", mode="auto")
    mw_result = mannwhitneyu(top_values, bottom_values, alternative="two-sided", method="auto")

    return {
        "ks_stat": float(ks_result.statistic),
        "ks_p_value": float(ks_result.pvalue),
        "mw_stat": float(mw_result.statistic),
        "mw_p_value": float(mw_result.pvalue),
    }


def _resolve_lopsided_metric_column(
    df: pd.DataFrame,
    lopsided_metric_column: str | None = None,
) -> str:
    """Resolve which lopsidedness metric column to use for Top/Bottom splitting."""
    if lopsided_metric_column is not None:
        _require_columns(df, [lopsided_metric_column], "Full analysis input")
        return lopsided_metric_column

    candidate_columns = [
        "Centroid_Offset",
        "centroid_offset",
        "Centroid_Distance",
        "centroid",
        "MRL",
        "mrl",
        "Pairwise",
        "pairwise",
        "Hemi_Contrast",
        "hemi_contrast",
        "Normalised_RMS_Distance",
        "normalised_rms_distance",
    ]
    for candidate in candidate_columns:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not resolve a lopsidedness metric column. Provide lopsided_metric_column explicitly."
    )


def _build_top_bottom_lopsided_flags(
    df: pd.DataFrame,
    metric_column: str,
    reference_snapshot: int = 99,
    split_fraction: float = 1 / 3,
    pairwise_cap: float = 90.0,
) -> pd.DataFrame:
    """Create Snap99 Top/Bottom lopsided flags from a selected metric column."""
    _require_columns(
        df,
        ["HaloID_Snap99", "SnapNum", metric_column],
        "Top/Bottom split input",
    )

    work_df = df.loc[:, ["HaloID_Snap99", "SnapNum", metric_column]].copy()
    work_df["SnapNum"] = pd.to_numeric(work_df["SnapNum"], errors="coerce")
    work_df[metric_column] = pd.to_numeric(work_df[metric_column], errors="coerce")

    snap_df = work_df.loc[work_df["SnapNum"] == reference_snapshot].copy()
    snap_df = snap_df.dropna(subset=["HaloID_Snap99", metric_column])
    snap_df = snap_df.drop_duplicates(subset=["HaloID_Snap99"], keep="first")

    if snap_df.empty:
        return pd.DataFrame(columns=["HaloID_Snap99", "lopsided_flag"])

    items = list(zip(snap_df["HaloID_Snap99"], snap_df[metric_column]))
    metric_name_norm = metric_column.strip().lower()

    if metric_name_norm == "pairwise":
        valid_items = [
            (halo_id, value)
            for halo_id, value in items
            if value is not None and not np.isnan(value) and value <= pairwise_cap
        ]
        # Preserve existing behavior: Pairwise is inverted for top/bottom mapping.
        valid_items.sort(key=lambda item: item[1], reverse=True)
    else:
        valid_items = [
            (halo_id, value)
            for halo_id, value in items
            if value is not None and not np.isnan(value)
        ]
        valid_items.sort(key=lambda item: item[1])

    if not valid_items:
        return pd.DataFrame(columns=["HaloID_Snap99", "lopsided_flag"])

    n_select = int(len(valid_items) * split_fraction)
    if n_select <= 0:
        return pd.DataFrame(columns=["HaloID_Snap99", "lopsided_flag"])

    bottom_slice = valid_items[:n_select]
    top_slice = valid_items[-n_select:]

    top_ids = {halo_id for halo_id, _ in top_slice}
    bottom_ids = {halo_id for halo_id, _ in bottom_slice}
    bottom_ids = bottom_ids - top_ids

    rows: list[dict[str, Any]] = []
    for halo_id in sorted(top_ids):
        rows.append({"HaloID_Snap99": halo_id, "lopsided_flag": "Top"})
    for halo_id in sorted(bottom_ids):
        rows.append({"HaloID_Snap99": halo_id, "lopsided_flag": "Bottom"})

    return pd.DataFrame(rows)


def run_full_analysis(
    df: pd.DataFrame,
    reference_snapshot: int = 99,
    lopsided_metric_column: str | None = None,
    distribution_feature: str = "mean_mag_gap",
    split_fraction: float = 1 / 3,
    pairwise_cap: float = 90.0,
) -> dict[str, Any]:
    """Run the end-to-end modular analysis pipeline.

    Required columns:
    - HaloID_Snap99
    - SnapNum
    - Mag_Gap
    - Merger_Class

    Pipeline:
    1. Build Snap99 Top/Bottom lopsided split from a selected metric.
    2. Run contingency analysis between lopsided_flag and Merger_Class.
    3. Compute halo-level merger features from magnitude-gap evolution.
    4. Compare Top vs Bottom feature distributions.

    Notes:
    - No plotting, file writing, or in-place mutation is performed.
    """
    _validate_input_dataframe(df)

    metric_column = _resolve_lopsided_metric_column(
        df=df,
        lopsided_metric_column=lopsided_metric_column,
    )

    lopsided_flags = _build_top_bottom_lopsided_flags(
        df=df,
        metric_column=metric_column,
        reference_snapshot=reference_snapshot,
        split_fraction=split_fraction,
        pairwise_cap=pairwise_cap,
    )

    class_df = (
        df.loc[:, ["HaloID_Snap99", "Merger_Class"]]
        .dropna(subset=["HaloID_Snap99", "Merger_Class"])
        .drop_duplicates(subset=["HaloID_Snap99"], keep="first")
        .copy()
    )

    contingency_input = lopsided_flags.merge(class_df, on="HaloID_Snap99", how="left")
    contingency_input = contingency_input.dropna(subset=["Merger_Class"])
    contingency = compute_contingency_analysis(contingency_input)

    features = compute_merger_features(df=df, reference_snapshot=reference_snapshot)
    features_with_flags = features.merge(
        lopsided_flags,
        on="HaloID_Snap99",
        how="left",
    )

    distribution_tests = compare_distributions(
        df=features_with_flags,
        feature_column=distribution_feature,
    )

    return {
        "contingency": contingency,
        "features": features_with_flags,
        "distribution_tests": distribution_tests,
    }
