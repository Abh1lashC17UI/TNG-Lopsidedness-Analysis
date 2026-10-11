"""extract_extremes.py — Snap99 Top/Bottom 5 halo extractor.

Usage
-----
If BASE_DIR is set below, run with no arguments:
    python extract_extremes.py
    python extract_extremes.py --top_10

If BASE_DIR is left as None, a path must be supplied on the command line:
    python extract_extremes.py halo_exports/TNG100-1
    python extract_extremes.py halo_exports/TNG50-1
    python extract_extremes.py halo_exports/TNG50-1 --top_10
    python extract_extremes.py halo_exports/TNG50-1/Snap99 --top_10

Output
------
Default mode:
    Writes  top_bottom_5_systems_snap99.csv  to the simulation root folder
    (e.g. halo_exports/TNG100-1/).

--top_10 mode:
    Writes  top_bottom_5_systems_snap99_top_10.csv  to the same folder,
    using *_metrics_top_10.csv files and restricting to halos with
    GroupNsubs >= 10.

Path resolution accepts either the simulation root or the Snap99/
sub-directory; the output is always written to the simulation root.

Sorting logic:
  - Pairwise  : values > 90.0 excluded; sorted ascending
                -> first 5  = Most Lopsided (angles furthest below 90)
                -> last  5  = Least Lopsided (angles closest to 90, ≤ 90)
  - All others: sorted ascending
                -> last  5  = Most Lopsided (largest values)
                -> first 5  = Least Lopsided (smallest values)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# Configuration -- set BASE_DIR here to skip the CLI argument requirement,
# or leave as None to require it at the command line.
BASE_DIR = r"C:\Users\Abhilash\Documents\Thesis\halo_exports\TNG100-1"
# BASE_DIR = r"C:\Users\Abhilash\Documents\Thesis\halo_exports\TNG50-1"

PAIRWISE_CAP: float = 90.0
N_EXTREMES: int = 5

# Canonical metric name -> lowercase CSV column name
METRICS: list[tuple[str, str]] = [
    ("Centroid_Offset", "centroid"),
    ("MRL",             "mrl"),
    ("Pairwise",        "pairwise"),
    ("Hemi_Contrast",   "hemi_contrast"),
]

# Column names for satellite-count lookup (checked in order)
NSAT_COLUMNS: tuple[str, ...] = ("GroupNsubs", "GroupNSubs", "subhalo_count")

COL_MOST = "Most Lopsided (Top 5) IDs"
COL_LEAST = "Least Lopsided (Bottom 5) IDs"


def _resolve_paths(raw_path: str) -> tuple[str, str]:
    """Resolve (sim_dir, snap99_dir) from a raw path.

    Accepts either the simulation root (contains Snap99/) or the Snap99/
    sub-directory itself. Always returns absolute paths.
    """
    target = os.path.abspath(raw_path)
    if os.path.basename(target).lower() == "snap99":
        sim_dir    = os.path.dirname(target)
        snap99_dir = target
    else:
        sim_dir    = target
        snap99_dir = os.path.join(sim_dir, "Snap99")
    return sim_dir, snap99_dir


def _parse_args() -> tuple[str, bool]:
    """Return (sim_dir, is_top_10) resolved from CLI args and script constants."""
    parser = argparse.ArgumentParser(
        description="Extract Top/Bottom 5 lopsided halos from Snap99 metric CSVs."
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=BASE_DIR,
        help=(
            "Path to the simulation export folder (or its Snap99/ sub-directory). "
            "Accepts e.g. halo_exports/TNG100-1 or halo_exports/TNG100-1/Snap99."
        ),
    )
    parser.add_argument(
        "--top_10",
        action="store_true",
        help=(
            "Restrict analysis to the top-10 satellite population using "
            "*_metrics_top_10.csv files and filtering for GroupNsubs >= 10."
        ),
    )
    args = parser.parse_args()
    if args.base_dir is None:
        parser.error("No BASE_DIR supplied on command line and script-level BASE_DIR is None.")
    return args.base_dir, args.top_10


def _collect_metric_files(snap99_dir: str, is_top_10: bool) -> list[str]:
    """Glob Snap99 metrics CSVs according to the selected mode.

    Default mode: *_metrics.csv (top_10 files excluded).
    Top-10 mode : *_metrics_top_10.csv only.
    """
    if is_top_10:
        pattern = os.path.join(snap99_dir, "Halo *", "*_metrics_top_10.csv")
        files = glob.glob(pattern)
        print(f"[top_10 mode] Found {len(files)} *_metrics_top_10.csv files.")
    else:
        pattern   = os.path.join(snap99_dir, "Halo *", "*_metrics.csv")
        all_files = glob.glob(pattern)
        files     = [f for f in all_files if "top_10" not in os.path.basename(f)]
        skipped   = len(all_files) - len(files)
        print(f"Found {len(all_files)} metrics CSVs -> kept {len(files)}, skipped {skipped} (top_10).")
    return files


def _parse_halo_id(filepath: str) -> int | None:
    """Extract HaloID from the parent folder name (e.g. 'Halo 552' -> 552)."""
    folder = os.path.basename(os.path.dirname(filepath))
    match = re.search(r"Halo\s+(\d+)", folder)
    if match:
        return int(match.group(1))
    return None


def _read_row(filepath: str) -> pd.Series:
    """Read the single data row from a metrics CSV."""
    return pd.read_csv(filepath).iloc[0]


def _read_sat_count(row: pd.Series) -> float:
    """Return the satellite count from a metrics row, or NaN if absent."""
    for col in NSAT_COLUMNS:
        val = row.get(col, np.nan)
        if not pd.isna(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return np.nan


def build_aggregated_dataframe(
    files: list[str],
    is_top_10: bool = False,
) -> pd.DataFrame:
    """Return one row per halo with HaloID + core metric columns.

    When is_top_10 is True, halos with GroupNsubs < 10 (or missing) are
    strictly excluded and a summary log is printed.
    """
    canonical_cols = [m[0] for m in METRICS]
    csv_cols       = [m[1] for m in METRICS]

    records: list[dict] = []
    seen_ids: set[int] = set()
    n_skipped_nsat = 0

    for filepath in files:
        halo_id = _parse_halo_id(filepath)
        if halo_id is None:
            print(f"  WARNING: could not parse HaloID from {filepath!r} -- skipping.")
            continue

        if halo_id in seen_ids:
            print(f"  WARNING: duplicate HaloID {halo_id} encountered ({filepath!r}) -- skipping.")
            continue
        seen_ids.add(halo_id)

        try:
            row = _read_row(filepath)
        except Exception as exc:
            print(f"  WARNING: failed to read {filepath!r}: {exc} -- skipping.")
            continue

        # GroupNsubs filter for top-10 mode
        if is_top_10:
            sat_count = _read_sat_count(row)
            if pd.isna(sat_count) or sat_count < 10:
                n_skipped_nsat += 1
                continue

        record: dict = {"HaloID": halo_id}
        for canonical, csv_col in zip(canonical_cols, csv_cols):
            val = row.get(csv_col, np.nan)

            # Mirrors _extract_metric_value in analysis.py
            if canonical == "Hemi_Contrast" and not pd.isna(val):
                val = abs(val)

            try:
                record[canonical] = float(val)
            except (TypeError, ValueError):
                record[canonical] = np.nan

        records.append(record)

    df = pd.DataFrame(records, columns=["HaloID"] + canonical_cols)

    if is_top_10:
        print(
            f"Found {len(files)} files -> Kept {len(df)} halos with GroupNsubs >= 10, "
            f"skipped {n_skipped_nsat} halos."
        )
    else:
        print(f"Aggregated {len(df)} halos successfully.")

    return df


def _extract_extremes(
    df: pd.DataFrame,
    metric: str,
    n: int,
    is_pairwise: bool,
) -> tuple[list[int], list[int]]:
    """Return (most_lopsided_ids, least_lopsided_ids) for one metric.

    Most extreme -> least extreme ordering is preserved within each list.
    Systems with Pairwise > 90.0 degrees are strictly excluded.
    """
    col = df[["HaloID", metric]].dropna(subset=[metric]).copy()

    if is_pairwise:
        # Explicitly filter out any systems where Pairwise angle > 90.0
        n_before = len(col)
        col = col[col[metric] <= PAIRWISE_CAP]
        n_excluded = n_before - len(col)
        print(
            f"    [Pairwise filter] Excluded {n_excluded} systems with Pairwise > "
            f"{PAIRWISE_CAP} deg ({len(col)} valid systems retained)."
        )
        assert (col[metric] <= PAIRWISE_CAP).all(), (
            f"Pairwise values > {PAIRWISE_CAP} detected after filtering!"
        )

        # Sort ascending: lowest angles (furthest from 90) -> Most Lopsided
        col_sorted = col.sort_values(metric, ascending=True)
        most_ids  = col_sorted["HaloID"].iloc[:n].astype(int).tolist()   # lowest = most lopsided
        least_ids = col_sorted["HaloID"].iloc[-n:].astype(int).tolist()  # closest to 90 = least lopsided
        # Reverse tail so closest-to-90 comes first within the bottom 5
        least_ids = least_ids[::-1]
    else:
        # Sort ascending: largest values at bottom -> Most Lopsided
        col_sorted = col.sort_values(metric, ascending=True)
        least_ids = col_sorted["HaloID"].iloc[:n].astype(int).tolist()
        most_ids  = col_sorted["HaloID"].iloc[-n:].astype(int).tolist()
        # most_ids comes from the tail -> reverse for most-extreme-first
        most_ids  = most_ids[::-1]

    return most_ids, least_ids


def main() -> None:
    raw_path, is_top_10 = _parse_args()

    sim_dir, snap99_dir = _resolve_paths(raw_path)

    if not os.path.isdir(sim_dir):
        print(f"ERROR: Simulation directory does not exist or is not a directory: {sim_dir!r}")
        sys.exit(1)

    if not os.path.isdir(snap99_dir):
        print(f"ERROR: Snap99 sub-directory not found: {snap99_dir!r}")
        sys.exit(1)

    print(f"Simulation root : {sim_dir}")
    print(f"Snap99 directory: {snap99_dir}")
    print(f"Mode            : {'top_10 satellites' if is_top_10 else 'all satellites'}")

    files = _collect_metric_files(snap99_dir, is_top_10)
    if not files:
        print("ERROR: No metrics CSVs found. Check the directory structure.")
        sys.exit(1)

    df = build_aggregated_dataframe(files, is_top_10=is_top_10)

    rows: list[dict] = []
    for canonical, _ in METRICS:
        is_pairwise = (canonical == "Pairwise")
        most_ids, least_ids = _extract_extremes(df, canonical, N_EXTREMES, is_pairwise)
        rows.append({
            "Metric":  canonical,
            COL_MOST:  ", ".join(map(str, most_ids)),
            COL_LEAST: ", ".join(map(str, least_ids)),
        })
        print(f"  {canonical:<28}  most={most_ids}  least={least_ids}")

    out_filename = (
        "top_bottom_5_systems_snap99_top_10.csv" if is_top_10
        else "top_bottom_5_systems_snap99.csv"
    )
    out_df   = pd.DataFrame(rows, columns=["Metric", COL_MOST, COL_LEAST])
    out_path = os.path.join(sim_dir, out_filename)

    try:
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved -> {os.path.abspath(out_path)}")
    except PermissionError:
        stem, ext = os.path.splitext(out_filename)
        fallback_path = os.path.join(sim_dir, f"{stem}_new{ext}")
        out_df.to_csv(fallback_path, index=False)
        print(f"\nWARNING: Could not overwrite {out_path!r} (it may be open in Excel).")
        print(f"Saved to fallback file instead -> {os.path.abspath(fallback_path)}")


if __name__ == "__main__":
    main()
