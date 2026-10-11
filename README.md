# IllustrisTNG Satellite Lopsidedness Analysis

Quantifying the spatial lopsidedness of satellite galaxy distributions within the **IllustrisTNG** cosmological simulations (**TNG50-1** and **TNG100-1**).

## Overview

This repository implements the analysis pipeline for a master's thesis studying the anisotropy of luminous satellite systems around Milky-Way-mass host halos ($50 \leq M_{\mathrm{Group}} \leq 500$ in simulation units). The project:

1. Extracts group catalogs from TNG50-1 and TNG100-1 across snapshots 68–99 (z ≈ 0.5 → 0).
2. Contrasts **unconstrained full censuses** against a **multiplicity-normalised Top-10 bright-satellite regime**, isolating the 10 most massive luminous satellites per host.
3. Evaluates four primary lopsidedness metrics against both **isotropic** (Monte-Carlo) and **cosmological** (empirical ensemble) null distributions:
   - **Centroid Offset (D_cen)**: Normalised displacement of the satellite centroid from the central galaxy.
   - **Mean Resultant Length (MRL)**: Directional clustering strength of satellite unit vectors.
   - **Pairwise Separation Angle (θ_lop)**: Mean angular separation between all satellite pairs.
   - **Hemispheric Contrast (C_hemi)**: Fractional excess in the most populated hemisphere.
4. Tracks temporal evolution of these metrics along progenitor lineages.
5. Correlates spatial asymmetry with assembly history, merger events, and magnitude-gap (Δm₁₂) evolution.

---

## Code & Architecture Provenance Legend

To maintain clear academic and developmental transparency, modules, classes, and primary functions are cataloged with their origin attribution:

- **\[Human\]** : Conceived, authored, and validated entirely by the author.
- **\[AI-Assisted\]** : Conceptualized or drafted by human author with AI-assisted debugging, refactoring, or expansion.
- **\[AI-Developed\]** : Implemented via AI agent based on prompt instructions.

---

## Project Structure

### Core Modules

| File | Description | Provenance |
|:-----|:------------|:-----------|
| **classes.py** | `Halo` and `Subhalo` object modelling. Handles group catalog ingestion, periodic boundary coordinate unwrapping, `@cacheable_method` pickle caching in `.halo_cache/`, volume-based satellite membership, Top-10 satellite slicing via `process_top_10()`, merger tree classification via `analyze_merger_history()`, and CSV export through `Halo.save()`. | `[AI-Assisted]` |
| **metrics.py** | Lopsidedness metric implementations as class-based evaluators with Numba-accelerated kernels and multiprocessing batch modes: `Centroid.eval()`, `MRL.eval()`, `Pairwise.eval()`, `Hemi.eval()`, `TOI.eval3d()`/`eval2d()`, `Conical.eval()`. | `[Human]` |
| **calc.py** | Mathematical helper functions: isotropic unit vector generation (`isovecs_3d`, `isovecs_2d`, `isovecs_sph`), single and multi-vector angle computation (2D/3D), satellite radial distribution toy models (`radial_model`, `sample_radial`), and the `list_dict` container utility. | `[Human]` |
| **fetch.py** | HTTP interface to the [IllustrisTNG public API](https://www.tng-project.org/data/): `get()` for JSON endpoints, `get_binary()` for SubLink merger tree HDF5 downloads, and `unwrap_positions()` for periodic boundary correction of historical tracks. | `[Human]` |

### Statistical Significance

| File | Description | Provenance |
|:-----|:------------|:-----------|
| **p_value_engine.py** | `IsotropicNullEngine` class managing the `.p_val_cache/` directory. Vectorised Monte-Carlo generation for isotropic nulls (MRL, Pairwise, C_hemi) cached by N_sat; on-the-fly D_cen null conditioned on actual satellite radii. `compute_p_iso()` and `compute_p_cosmo()` produce per-metric p-values. `build_cosmo_ensemble()` aggregates snapshot-level metrics into empirical reference distributions. | `[AI-Assisted]` |
| **analysis.py** | Pure DataFrame-centric analysis utilities (no I/O, no UI): `aggregate_satellite_metric_rows()`, `aggregate_host_metric_rows()`, `compute_percentile_band()`, `process_raw_overview_data()`, `compute_correlation_matrices()`, `compute_contingency_analysis()` (χ², Cramér's V, standardised residuals), `compute_merger_features()`, `compare_distributions()` (KS, Mann-Whitney U), and `run_full_analysis()` end-to-end pipeline. | `[AI-Assisted]` |

### Interactive Visualisation & Reader

| File | Description | Provenance |
|:-----|:------------|:-----------|
| **reader.py** | Full-featured **PyQt5 interactive analysis suite** (≈5000 lines). Multi-tab interface for exploring individual halo systems and population-level statistics: | `[AI-Assisted]` |

**Reader GUI Tabs:**

| Tab | Class | Description | Provenance |
|:----|:------|:------------|:-----------|
| **3D View** | `Plot3DCanvas` | Interactive 3D scatter of satellite positions with camera presets (XY/XZ/YZ/Iso), selective subhalo and track rendering, virial radius sphere overlay, pick/hover interactivity, and `export_projections_white_bg()` for 1990×1990 publication-quality PNG export. | `[Human]` |
| **Temporal Lineage** | `MetricEvolutionCanvas` | Time-series plots of metric evolution along tracked progenitor lineages with snapshot slider navigation. | `[Human]` |
| **Overview: Satellite-Centric** | `MetricOverviewCanvas` | Lagrangian fixed-cohort median ± percentile bands across all tracked systems. | `[AI-Assisted]` |
| **Overview: Host-Centric** | `MetricOverviewCanvas` | Eulerian aperture-based per-snapshot aggregation of the evolving population. | `[AI-Assisted]` |
| **Statistical Significance** | `PValueTab` | Single-snapshot isotropic and cosmological p-values with lazy/on-demand calculation and `busy_state()` event pumping for UI responsiveness. | `[AI-Assisted]` |
| **Tracked Temporal p-values** | `SatCentricTab` | Temporal p-value trajectories along tracked progenitor histories. | `[AI-Assisted]` |
| **System Ranking Table** | `MetricSystemTable` | Sortable per-metric halo ranking table with satellite count histogram and CSV export. | `[Human]` |
| **Potential Reasons** | `PotentialReasonsView` | Merger history tracker, magnitude gap (Δm₁₂) evolution plotter, case-study image gallery, and correlation contingency heatmaps (χ², Cramér's V, standardised residuals). | `[AI-Assisted]` |

### Pipeline Entry Points & Data Processing

| File | Description | Provenance |
|:-----|:------------|:-----------|
| **tng_50_1.py** | Pipeline driver for TNG50-1: loads group catalogs from external storage, constructs `Halo` objects via volume-based satellite membership, applies mass cuts (50 ≤ M_central ≤ 300), and triggers batch metric computation and export. | `[Human]` |
| **tng_100_1.py** | Pipeline driver for TNG100-1 (same structure as above). | `[Human]` |
| **export_specific_halos.py** | Interactive utility for targeted per-halo exports across multiple snapshots. Reads `halo_lineage_summary.csv`, identifies un-exported halos, and provides a menu-driven interface for selective or batch export. | `[AI-Assisted]` |
| **process_top_10.py** | Standalone batch processor: reads `*_positions.csv` files, isolates the 10 most massive satellites, recalculates metrics, re-filters tracks, and produces all `*_top_10.csv` variants. | `[AI-Assisted]` |
| **generate_top10_list.py** | Scans export directories for existing Top-10 processed systems and produces `top_10_systems_summary.csv`. | `[Human]` |

### Lineage, Ranking & Merger Analysis

| File | Description | Provenance |
|:-----|:------------|:-----------|
| **interesting_halo_finder.py** | Builds `halo_lineage_summary.csv`: reads central subhalo tracks from Snap99 exports, maps SubfindIDs → HaloIDs across all snapshots, and records export completion status. | `[Human]` |
| **extract_ranks.py** | Produces `system_ranks_snap99.csv` / `system_ranks_snap99_top_10.csv`: per-metric ranking of all Snap99 halos (Rank 1 = most lopsided), with Pairwise > 90° filtering and `--top_10` mode. | `[AI-Assisted]` |
| **extract_extremes.py** | Produces `top_bottom_5_systems_snap99.csv` / `top_bottom_5_systems_snap99_top_10.csv`: identifies the 5 most and 5 least lopsided halos per metric at Snap99. | `[AI-Developed]` |
| **potential_merger_analysis.py** | Step-detection algorithm for sustained satellite count changes (Infall/Loss events) using configurable window sizes, absolute change thresholds, and cooldown periods. Produces comparative plots of lopsided vs. isotropic system histories. | `[AI-Assisted]` |

### Utilities

| File | Description | Provenance |
|:-----|:------------|:-----------|
| **halo_data_folder_organiser.py** | Sorts flat CSV exports into `Halo <ID>/` subdirectories using regex extraction of halo IDs from filenames. | `[Human]` |
| **get_snaps.py** | Parallel snapshot downloader using `wget` against the TNG API, with SHA-256 checksum verification. | `[AI-Assisted]` |
| **watcher.py** | Remote monitoring script: periodically screenshots terminal windows running long extraction tasks and saves timestamped PNG captures. | `[AI-Assisted]` |

---

## Requirements & Installation

| Package | Minimum Version |
|:--------|:---------------|
| Python | ≥ 3.9 |
| PyQt5 | ≥ 5.15 |
| numpy | — |
| scipy | — |
| pandas | — |
| matplotlib | — |
| seaborn | — |
| h5py | — |
| illustris_python | — |
| tqdm | — |

```bash
pip install PyQt5 numpy scipy pandas matplotlib seaborn h5py tqdm
# illustris_python: https://github.com/illustristng/illustris_python
```

---

## Usage

### 1. Data Extraction

Run the simulation-specific pipeline drivers (requires local access to TNG group catalogs):

```bash
python tng_50_1.py
python tng_100_1.py
```

For targeted per-halo export across snapshots:

```bash
python export_specific_halos.py
```

### 2. Post-Processing: Top-10 Satellite Selection

```bash
python process_top_10.py
```

### 3. Lineage & Ranking

```bash
python interesting_halo_finder.py
python extract_ranks.py [--top_10]
python extract_extremes.py [--top_10]
```

### 4. Interactive Analysis (The Reader)

```bash
python reader.py
```

- Select a simulation folder (e.g., `halo_exports/TNG100-1`)
- Browse lineage tracks, view 3D projections, examine metric evolution
- Compute statistical significance (isotropic and cosmological p-values)
- Explore merger history correlations via the Potential Reasons tab
- Export publication-quality figures via white-background export buttons

### 5. Merger Analysis

```bash
python potential_merger_analysis.py
```

---

## Directory Hierarchy

```
.
├── .halo_cache/                        # Pickled Halo objects (auto-managed)
├── .p_val_cache/                       # Root-level isotropic null cache (legacy)
├── halo_exports/
│   ├── TNG100-1/
│   │   ├── .p_val_cache/               # Per-simulation isotropic null distribution cache
│   │   ├── interesting_systems/        # Case-study figures (full satellite population)
│   │   │   ├── Halo943/
│   │   │   ├── Halo1429/
│   │   │   └── Halo1639/
│   │   ├── interesting_systems_top_10/ # Case-study figures (Top-10 regime)
│   │   │   ├── Halo575/
│   │   │   ├── Halo847/
│   │   │   ├── Halo1185/
│   │   │   └── Halo1507/
│   │   ├── merger_plots/               # Merger analysis output figures
│   │   ├── metrics_overview_plots/     # Population overview exported figures
│   │   ├── Snap68/                     # Earliest exported snapshot
│   │   │   ├── Halo 1000/
│   │   │   │   ├── sim_TNG100-1_halo_1000_snap_68_positions.csv
│   │   │   │   ├── sim_TNG100-1_halo_1000_snap_68_positions_top_10.csv
│   │   │   │   ├── sim_TNG100-1_halo_1000_snap_68_dark_positions.csv
│   │   │   │   ├── sim_TNG100-1_halo_1000_snap_68_metrics.csv
│   │   │   │   ├── sim_TNG100-1_halo_1000_snap_68_metrics_top_10.csv
│   │   │   │   ├── sim_TNG100-1_halo_1000_snap_68_all_tracks.csv
│   │   │   │   ├── sim_TNG100-1_halo_1000_snap_68_all_tracks_top_10.csv
│   │   │   │   ├── sim_TNG100-1_halo_1000_snap_68_central_ids.csv
│   │   │   │   ├── sim_TNG100-1_halo_1000_snap_68_metric_evolution.csv
│   │   │   │   ├── sim_TNG100-1_halo_1000_snap_68_metric_evolution_top_10.csv
│   │   │   │   ├── sim_TNG100-1_halo_1000_snap_68_merger_history.csv
│   │   │   │   └── sim_TNG100-1_halo_1000_snap_68_magnitude_gap.csv
│   │   │   └── ...
│   │   ├── Snap69/ ... Snap99/
│   │   ├── halo_lineage_summary.csv
│   │   ├── system_ranks_snap99.csv
│   │   ├── system_ranks_snap99_top_10.csv
│   │   ├── top_bottom_5_systems_snap99.csv
│   │   ├── top_bottom_5_systems_snap99_top_10.csv
│   │   └── top_10_systems.csv
│   └── TNG50-1/
│       └── (same structure as TNG100-1)
├── Writing/                            # Thesis manuscript and reference papers
├── lopsidedness/                       # Python virtual environment
├── classes.py
├── metrics.py
├── calc.py
├── fetch.py
├── p_value_engine.py
├── analysis.py
├── reader.py
├── tng_50_1.py
├── tng_100_1.py
├── export_specific_halos.py
├── process_top_10.py
├── generate_top10_list.py
├── interesting_halo_finder.py
├── extract_ranks.py
├── extract_extremes.py
├── potential_merger_analysis.py
├── halo_data_folder_organiser.py
├── get_snaps.py
├── watcher.py
├── README.md
├── data_layout.md
└── data_contract.md
```

---

## Key Classes & Functions Reference

### `classes.py`

| Item | Description | Provenance |
|:-----|:------------|:-----------|
| `@cacheable_method` | Decorator for `Halo` methods supporting pickle cache with `force_rerun` kwarg | `[AI-Assisted]` |
| `Halo.__new__()` | Singleton-style cache lookup: returns cached instance if `.halo_cache/` entry exists | `[AI-Assisted]` |
| `Halo.__init__()` | Initialises from raw group catalog data; skips if loaded from cache | `[AI-Assisted]` |
| `Halo.get_subhalos()` | Group-catalog-based satellite membership | `[Human]` |
| `Halo.get_subhalos_by_volume()` | Volume/virial-radius-based satellite membership with PBC unwrapping | `[Human]` |
| `Halo.calc_metrics()` | Computes all four core metrics for the current satellite population | `[Human]` |
| `Halo.track_visibile_subhalos()` | Progenitor tracking via SubLink merger trees (API or local) | `[Human]` |
| `Halo.process_top_10()` | Isolates 10 most massive satellites and recalculates metrics | `[Human]` |
| `Halo.calc_top_10_metrics_over_time()` | Metric evolution for the Top-10 tracked cohort | `[Human]` |
| `Halo.analyze_merger_history()` | Merger tree classification (Stable/Infall/Fly-by/Loss) with magnitude gap | `[Human]` |
| `Halo.save()` | CSV export orchestrator with per-product flags | `[AI-Assisted]` |
| `Subhalo` | Thin wrapper for individual subhalo properties, visibility checks, stellar mass | `[Human]` |

### `p_value_engine.py`

| Item | Description | Provenance |
|:-----|:------------|:-----------|
| `IsotropicNullEngine` | Manages `.p_val_cache/` pickle, stores sorted null distributions keyed by N_sat | `[AI-Assisted]` |
| `compute_p_iso()` | Per-metric isotropic p-values via `searchsorted` on cached nulls | `[AI-Assisted]` |
| `compute_p_cosmo()` | Per-metric cosmological p-values from empirical ensemble rankings | `[Human]` |
| `build_cosmo_ensemble()` | Snapshot-level metrics aggregation into reference DataFrame | `[AI-Developed]` |

### `reader.py` — GUI Classes

| Class | Description | Provenance |
|:------|:------------|:-----------|
| `HaloReaderApp` | Main application widget: folder selection, lineage navigation, tab management, Top-10 toggle, `busy_state()` context manager with throttled `pump_events()` | `[AI-Assisted]` |
| `Plot3DCanvas` | 3D Matplotlib FigureCanvas with dark/light theme switching and multi-projection export | `[Human]` |
| `MetricEvolutionCanvas` | 2D time-series FigureCanvas for per-halo metric evolution | `[AI-Assisted]` |
| `MetricOverviewCanvas` | Population-level overview canvas (satellite-centric and host-centric modes) | `[AI-Assisted]` |
| `PValueCanvas` | Null distribution histograms and temporal p-value trajectory plots | `[AI-Developed]` |
| `PValueTab` | Widget combining PValueCanvas with metric summary table | `[AI-Developed]` |
| `SatCentricTab` | Widget for satellite-centric tracked temporal p-value trajectories | `[AI-Assisted]` |
| `MetricSystemTable` | Sortable system ranking table with satellite count histogram | `[AI-Assisted]` |
| `PotentialReasonsView` | Multi-tab merger analysis viewer: gallery, magnitude-gap plotter, correlation heatmaps | `[AI-Assisted]` |

---

## License

*To be determined.*
