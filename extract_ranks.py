"""extract_ranks.py — Snap99 complete lopsidedness ranking for all halos.

Usage
-----
If BASE_DIR is set below, run with no arguments:
    python extract_ranks.py
    python extract_ranks.py --top_10

If BASE_DIR is left as None, a path must be supplied on the command line:
    python extract_ranks.py halo_exports/TNG100-1
    python extract_ranks.py halo_exports/TNG50-1
    python extract_ranks.py halo_exports/TNG50-1 --top_10
    python extract_ranks.py halo_exports/TNG50-1/Snap99 --top_10

Output
------
Default mode:
    Writes  system_ranks_snap99.csv  to the simulation root folder
    (e.g. halo_exports/TNG100-1/).

--top_10 mode:
    Writes  system_ranks_snap99_top_10.csv  to the same folder,
    using *_metrics_top_10.csv files and restricting to halos with
    GroupNsubs >= 10.

Path resolution accepts either the simulation root or the Snap99/
sub-directory; the output is always written to the simulation root.

Ranking logic (Rank 1 = most lopsided):
  - Pairwise              : values > 90.0 are set to NaN and ranked to the
                            bottom; remaining ranked ascending (smallest = Rank 1).
  - Pairwise_Exceeds_90   : boolean flag placed directly after Pairwise_Rank
                            identifying anomalous systems where Pairwise > 90.0.
  - All other metrics     : ranked descending (largest value = Rank 1).
  - Ties                  : broken by method='min' (tied halos share the
                            lowest rank in the group).
  - Missing / NaN values  : pushed to the bottom via na_option='bottom'.
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
# BASE_DIR = r"C:\Users\Abhilash\Documents\Thesis\halo_exports\TNG100-1"
BASE_DIR = r"C:\Users\Abhilash\Documents\Thesis\halo_exports\TNG50-1"

PAIRWISE_CAP: float = 90.0

# Canonical metric name -> lowercase CSV column name
METRICS: list[tuple[str, str]] = [
    ("Centroid_Offset",         "centroid"),
    ("MRL",                     "mrl"),
    ("Pairwise",                "pairwise"),
    ("Hemi_Contrast",           "hemi_contrast"),
    ("Normalised_RMS_Distance", "normalised_rms_distance"),
]

# Column names for satellite-count lookup (checked in order)
NSAT_COLUMNS: tuple[str, ...] = ("GroupNsubs", "GroupNSubs", "subhalo_count")


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
        description="Rank all Snap99 halos by lopsidedness from metric CSVs."
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


def compute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add a rank column for each metric; Rank 1 = most lopsided.

    Ranking rules:
    - Pairwise: values > PAIRWISE_CAP are set to NaN before ranking;
                remaining values ranked ascending (smallest = Rank 1).
    - Pairwise_Exceeds_90: boolean flag placed directly after Pairwise_Rank.
    - All others: ranked descending (largest = Rank 1).
    - Ties resolved by method='min' (tied halos share lowest rank in group).
    - NaN / invalid values receive the worst (bottom) rank via na_option='bottom'.

    Returns a new DataFrame with HaloID, rank columns, and Pairwise_Exceeds_90.
    """
    df = df.copy()
    df["Pairwise_Exceeds_90"] = (df["Pairwise"] > PAIRWISE_CAP).fillna(False)

    ranked = df[["HaloID"]].copy()

    for canonical, _ in METRICS:
        rank_col    = f"{canonical}_Rank"
        is_pairwise = (canonical == "Pairwise")

        if is_pairwise:
            # Cap: treat values > PAIRWISE_CAP as missing so they sink to bottom
            capped = df[canonical].where(df[canonical] <= PAIRWISE_CAP)
            ranked[rank_col] = (
                capped
                .rank(method="min", ascending=True, na_option="bottom")
                .astype(int)
            )
            # Insert the boolean flag directly after Pairwise_Rank
            ranked["Pairwise_Exceeds_90"] = df["Pairwise_Exceeds_90"].values
        else:
            ranked[rank_col] = (
                df[canonical]
                .rank(method="min", ascending=False, na_option="bottom")
                .astype(int)
            )

    return ranked


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

    df_raw = build_aggregated_dataframe(files, is_top_10=is_top_10)

    print("\nComputing ranks...")
    df_ranks = compute_ranks(df_raw)
    df_ranks = df_ranks.sort_values("HaloID").reset_index(drop=True)

    out_filename = (
        "system_ranks_snap99_top_10.csv" if is_top_10
        else "system_ranks_snap99.csv"
    )
    out_path = os.path.join(sim_dir, out_filename)

    try:
        df_ranks.to_csv(out_path, index=False)
        print(f"Saved -> {os.path.abspath(out_path)}")
    except PermissionError:
        stem, ext = os.path.splitext(out_filename)
        fallback_path = os.path.join(sim_dir, f"{stem}_new{ext}")
        df_ranks.to_csv(fallback_path, index=False)
        print(f"\nWARNING: Could not overwrite {out_path!r} (it may be open in Excel).")
        print(f"Saved to fallback file instead -> {os.path.abspath(fallback_path)}")

    print(f"\nRank table shape: {df_ranks.shape[0]} halos x {df_ranks.shape[1]} columns")
    print(df_ranks.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
