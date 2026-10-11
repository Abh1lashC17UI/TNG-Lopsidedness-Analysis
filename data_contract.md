# Data Contract

This document defines the schema, naming, reference frame, fallback, and validation contracts shared across the three pipeline layers:

- **Data Generation** (`classes.py`, `tng_50_1.py`, `tng_100_1.py`, `process_top_10.py`) `[PENDING: Human | AI-Assisted | AI-Developed]`
- **Analysis** (`analysis.py`, `p_value_engine.py`, `extract_ranks.py`, `extract_extremes.py`, `potential_merger_analysis.py`) `[PENDING: Human | AI-Assisted | AI-Developed]`
- **Visualisation** (`reader.py`) `[PENDING: Human | AI-Assisted | AI-Developed]`

---

## Code & Architecture Provenance Legend

To maintain clear academic and developmental transparency, modules, classes, and primary functions are cataloged with their origin attribution:

- **\[Human\]** : Conceived, authored, and validated entirely by the author.
- **\[AI-Assisted\]** : Conceptualized or drafted by human author with AI-assisted debugging, refactoring, or expansion.
- **\[AI-Developed\]** : Implemented via AI agent based on prompt instructions.

> *Note: Status tags are pre-populated as `[PENDING: Human | AI-Assisted | AI-Developed]` for final author review.*

---

## File Naming Contract

### Standard Pattern

```
sim_<Simulation>_halo_<HaloID>_snap_<Snapshot>_<type>.csv
```

**Examples:**
- `sim_TNG100-1_halo_552_snap_99_metrics.csv`
- `sim_TNG100-1_halo_552_snap_99_positions.csv`
- `sim_TNG100-1_halo_552_snap_99_dark_positions.csv`
- `sim_TNG100-1_halo_552_snap_99_all_tracks.csv`
- `sim_TNG100-1_halo_552_snap_99_central_ids.csv`
- `sim_TNG100-1_halo_552_snap_99_metric_evolution.csv`
- `sim_TNG100-1_halo_552_snap_99_merger_history.csv`
- `sim_TNG100-1_halo_552_snap_99_magnitude_gap.csv`

### Top-10 Variant Pattern

```
sim_<Simulation>_halo_<HaloID>_snap_<Snapshot>_<type>_top_10.csv
```

**Examples:**
- `sim_TNG100-1_halo_552_snap_99_positions_top_10.csv`
- `sim_TNG100-1_halo_552_snap_99_metrics_top_10.csv`
- `sim_TNG100-1_halo_552_snap_99_all_tracks_top_10.csv`
- `sim_TNG100-1_halo_552_snap_99_metric_evolution_top_10.csv`

---

## Reference Frame & Units Convention

| Quantity | Reference Frame / Unit |
|:---------|:----------------------|
| **Positions** (`x`, `y`, `z`) | Relative to the **central subhalo** position (`SubhaloPos[0]`), in **comoving kpc/h**, with periodic boundary unwrapping applied using simulation box size (L_Box = 75,000 ckpc/h for TNG100-1; 35,000 ckpc/h for TNG50-1). Central is at `(0, 0, 0)`. |
| **Absolute Central Position** (`CentralPos_x/y/z`) | Absolute box coordinates in **comoving kpc/h** within `[0, BoxSize)`. |
| **Group_R_Crit200** | Virial radius $R_{200\mathrm{c}}$ in **comoving kpc/h**. |
| **GroupMass**, **SubhaloMass**, **Central_Galaxy_Mass** | Simulation mass units: $10^{10} M_\odot / h$. |
| **Centroid Offset (`centroid`)** | Dimensionless: $\|\bar{\mathbf{r}}\| / r_{\mathrm{rms}}$, normalised by root-mean-square satellite distance. |
| **MRL (`mrl`)** | Dimensionless in $[0, 1]$: $\|\frac{1}{N}\sum \hat{\mathbf{r}}_i\|$. |
| **Pairwise (`pairwise`)** | **Degrees** in $[0°, 180°]$: mean angular separation across all satellite pairs. Isotropic expectation = 90°. Lower values = more clustered/lopsided. |
| **Hemispheric Contrast (`hemi_contrast`)** | Dimensionless in $[0, 1]$: $\max_{\hat{\mathbf{n}}} |N_+ - N_-| / N$ evaluated over an isotropic grid (`num_grid = 30`). |
| **Normalised RMS Distance** | Dimensionless: $r_{\mathrm{rms}} / R_{200\mathrm{c}}$. |
| **Photometric Magnitudes** | Absolute AB magnitudes from `SubhaloStellarPhotometrics` (8 bands: U, B, V, K, g, r, i, z; r-band at index 5 is used for magnitude gap). |

---

## Core File Schema Specifications

### 1. Metrics File (`*_metrics.csv`) `[PENDING: Human | AI-Assisted | AI-Developed]`

Single-row CSV containing scalar halo-level properties and metric values.

| Column | Type | Unit | Description |
|:-------|:-----|:-----|:------------|
| `GroupCM` | str (3-vector) | ckpc/h | Group center of mass |
| `GroupPos` | str (3-vector) | ckpc/h | Group position (minimum potential) |
| `GroupFirstSub` | int | — | SubfindID of the central subhalo |
| `GroupNsubs` | int | — | Total subhalos in FoF group |
| `GroupMass` | float | $10^{10} M_\odot/h$ | Total FoF group mass |
| `Group_R_Crit200` | float | ckpc/h | Virial radius $R_{200\mathrm{c}}$ |
| `centroid` | float | dimensionless | Normalised centroid shift (visible satellites) |
| `mrl` | float | dimensionless | Mean Resultant Length (visible satellites) |
| `pairwise` | float | degrees | Mean pairwise angular separation |
| `hemi_contrast` | float | dimensionless | Maximum hemispheric contrast |
| `dark_centroid` | float | dimensionless | Centroid shift including dark subhalos |
| `dark_mrl` | float | dimensionless | MRL including dark subhalos |
| `dark_pairwise` | float | degrees | Pairwise angle including dark subhalos |
| `dark_hemi_contrast` | float | dimensionless | Hemispheric contrast including dark subhalos |
| `subhalo_count` | int | — | Total subhalo count (visible + dark within volume) |
| `visible_subhalos` | int | — | Luminous satellite count |
| `normalised_rms_distance` | float | dimensionless | $r_{\mathrm{rms}} / R_{200\mathrm{c}}$ |
| `Central_Galaxy_Mass` | float | $10^{10} M_\odot/h$ | Mass of central subhalo |
| `Central_Galaxy_BH` | float | Sim units | Central BH accretion rate |
| `Central_Galaxy_Vel` | str (3-vector) | km/s | Central subhalo peculiar velocity |

### 2. Top-10 Metrics File (`*_metrics_top_10.csv`) `[PENDING: Human | AI-Assisted | AI-Developed]`

Single-row CSV generated by `Halo.process_top_10()` or `process_top_10.py`.

| Column | Type | Unit | Description |
|:-------|:-----|:-----|:------------|
| `centroid` | float | dimensionless | Centroid shift of Top-10 satellites |
| `mrl` | float | dimensionless | MRL of Top-10 satellites |
| `pairwise` | float | degrees | Pairwise angle of Top-10 satellites |
| `hemi_contrast` | float | dimensionless | Hemispheric contrast of Top-10 satellites |
| `normalised_rms_distance` | float | dimensionless | $r_{\mathrm{rms}} / R_{200\mathrm{c}}$ of Top-10 |
| `subhalo_count` | int | — | Number of satellites in subset (≤ 10) |
| `HaloID` | int | — | Host halo identifier |
| `SnapNum` | int | — | Snapshot number |

> **IMPORTANT — Fallback Contract**: `*_metrics_top_10.csv` does **not** carry `Group_R_Crit200`, `GroupMass`, or `visible_subhalos`. Any consumer needing these columns in Top-10 mode **must** fall back to the companion full `*_metrics.csv` file in the same directory (implemented in `HaloReaderApp._get_metrics_from_files()` and `HaloReaderApp.read_single_snap_metrics()`).

### 3. Positions File (`*_positions.csv`, `*_positions_top_10.csv`) `[PENDING: Human | AI-Assisted | AI-Developed]`

One row per visible subhalo. **Row 0 is always the central subhalo** at `(0, 0, 0)`; rows `1..N` are luminous satellites.

| Column | Type | Unit | Description |
|:-------|:-----|:-----|:------------|
| `x` | float | ckpc/h | X position relative to central |
| `y` | float | ckpc/h | Y position relative to central |
| `z` | float | ckpc/h | Z position relative to central |
| `subhalo_index` | int | — | Global SubfindID in snapshot group catalog |
| `SubhaloFlag` | bool/int | — | Cosmological origin flag (1 = genuine, 0 = fragment) |
| `SubhaloBHMdot` | float | Sim units | Black hole accretion rate |
| `SubhaloPos` | str (3-vector) | ckpc/h | Raw absolute box coordinates |
| `SubhaloParent` | int | — | Parent subhalo index |
| `SubhaloGrNr` | int | — | Parent FoF group number |
| `SubhaloMass` | float | $10^{10} M_\odot/h$ | Total subhalo mass |
| `SubhaloStellarPhotometrics` | str (8-vector) | AB mag | 8-band stellar photometrics |
| `SubhaloStarMetalFractions` | str (vector) | — | Stellar metal mass fractions |
| `SubhaloVel` | str (3-vector) | km/s | Peculiar velocity |

### 4. Dark Positions File (`*_dark_positions.csv`) `[PENDING: Human | AI-Assisted | AI-Developed]`

Same schema as `*_positions.csv`. Contains dark (non-luminous) subhalos within the search volume, relative to the central subhalo.

### 5. All Tracks File (`*_all_tracks.csv`, `*_all_tracks_top_10.csv`) `[PENDING: Human | AI-Assisted | AI-Developed]`

Concatenated historical trajectories for tracked visible satellites.

| Column | Type | Unit | Description |
|:-------|:-----|:-----|:------------|
| `x` | float | ckpc/h | Historical X position relative to central at that epoch |
| `y` | float | ckpc/h | Historical Y position relative to central at that epoch |
| `z` | float | ckpc/h | Historical Z position relative to central at that epoch |
| `subhalo_index` | int | — | **0-based row number** from the companion `*_positions.csv` (1 = first satellite, etc.) |

> **Ordering Contract**: Within each `subhalo_index` group, rows are ordered chronologically (oldest → newest, or newest at end). No explicit `SnapNum` column is present; time step alignment is performed from the end of each array backwards from the current snapshot.

### 6. Central IDs File (`*_central_ids.csv`) `[PENDING: Human | AI-Assisted | AI-Developed]`

Tracks the central subhalo's identity and absolute box position across snapshots.

| Column | Type | Unit | Description |
|:-------|:-----|:-----|:------------|
| `SnapNum` | int | — | Snapshot number |
| `SubfindID` | int | — | Central subhalo's global SubfindID at `SnapNum` |
| `BoxSize` | float | ckpc/h | Simulation box edge length |
| `CentralPos_x` | float | ckpc/h | Central absolute X coordinate |
| `CentralPos_y` | float | ckpc/h | Central absolute Y coordinate |
| `CentralPos_z` | float | ckpc/h | Central absolute Z coordinate |

### 7. Metric Evolution File (`*_metric_evolution.csv`, `*_metric_evolution_top_10.csv`) `[PENDING: Human | AI-Assisted | AI-Developed]`

Time series of lopsidedness metrics computed across the tracked satellite cohort.

| Column | Type | Unit | Description |
|:-------|:-----|:-----|:------------|
| `SnapNum` | int | — | Snapshot number (ascending order) |
| `Centroid_Distance` | float | dimensionless | Normalised centroid offset at `SnapNum` |
| `MRL` | float | dimensionless | Mean Resultant Length at `SnapNum` |
| `Pairwise` | float | degrees | Mean pairwise angle at `SnapNum` |
| `Hemi_Contrast` | float | dimensionless | Hemispheric contrast at `SnapNum` |
| `Normalised_RMS_Distance` | float | dimensionless | $r_{\mathrm{rms}} / R_{200\mathrm{c}}$ at `SnapNum` |

### 8. Merger History File (`*_merger_history.csv`) `[PENDING: Human | AI-Assisted | AI-Developed]`

Generated by `Halo.analyze_merger_history()` via `Halo.save(save_merger_analysis=True)`.

| Column | Type | Unit | Description |
|:-------|:-----|:-----|:------------|
| `SnapNum` | int | — | Snapshot number |
| `Mag_Central` | float | AB mag (r-band) | Central galaxy r-band absolute magnitude |
| `Mag_Secondary` | float | AB mag (r-band) | Second-brightest galaxy r-band magnitude |
| `Delta_m12` | float | mag | Magnitude gap $\Delta m_{12} = m_2 - m_1 \geq 0$ |
| `Merger_Class` | str | — | Interaction classification: `"Stable"`, `"Infall"`, `"Fly-by"`, or `"Loss"` |

### 9. Magnitude Gap File (`*_magnitude_gap.csv`) `[PENDING: Human | AI-Assisted | AI-Developed]`

Companion export containing the per-snapshot magnitude gap time series used by `PotentialReasonsView` and `analysis.compute_merger_features()`.

---

## Summary & Ranking File Contracts

### `halo_lineage_summary.csv` `[PENDING: Human | AI-Assisted | AI-Developed]`

- **Index**: `HaloID_at_Snap99` (int)
- **Columns**: `Halo_Category_Snap99` (str, value `"Tracked"`), followed by triplets per snapshot `XX` in `68..99`:
  - `HaloID_Snap_XX` (int, `-1` if unmapped)
  - `SubfindID_Snap_XX` (int, `-1` if missing)
  - `Exported_Snap_XX` (bool)

### `system_ranks_snap99.csv` / `system_ranks_snap99_top_10.csv` `[PENDING: Human | AI-Assisted | AI-Developed]`

| Column | Type | Description |
|:-------|:-----|:------------|
| `HaloID` | int | Snap99 Halo ID |
| `Centroid_Offset_Rank` | int | Rank (1 = largest offset = most lopsided) |
| `MRL_Rank` | int | Rank (1 = largest MRL = most lopsided) |
| `Pairwise_Rank` | int | Rank (1 = smallest angle ≤ 90° = most lopsided) |
| `Pairwise_Exceeds_90` | bool | Flag for anomalous systems with Pairwise > 90° |
| `Hemi_Contrast_Rank` | int | Rank (1 = largest contrast = most lopsided) |
| `Normalised_RMS_Distance_Rank` | int | Rank (1 = largest normalised RMS) |

### `top_bottom_5_systems_snap99.csv` / `top_bottom_5_systems_snap99_top_10.csv` `[PENDING: Human | AI-Assisted | AI-Developed]`

| Column | Type | Description |
|:-------|:-----|:------------|
| `Metric` | str | Canonical metric name (`Centroid_Offset`, `MRL`, `Pairwise`, `Hemi_Contrast`) |
| `Most Lopsided (Top 5) IDs` | str | Comma-separated Halo IDs (most extreme first) |
| `Least Lopsided (Bottom 5) IDs` | str | Comma-separated Halo IDs (most isotropic first) |

---

## Fallback & Validation Contracts

### 1. Top-10 Fallback Chain

1. **Group_R_Crit200**: Not present in `*_metrics_top_10.csv`. Consumers must locate the companion `*_metrics.csv` in the same `Halo <ID>/` folder and read `Group_R_Crit200` from it.
2. **Satellite count gating**: Top-10 mode requires `visible_subhalos >= 10` (or `GroupNsubs >= 10`). Halos with fewer than 10 luminous satellites must be excluded from Top-10 reference ensembles (`p_value_engine.get_cosmological_ensemble(top10=True)`) and have their Top-10 UI checkboxes disabled (`HaloReaderApp._gate_top10_checkboxes()`).
3. **Column name aliases**: Metric columns appear in both lowercase (`centroid`, `mrl`, `pairwise`, `hemi_contrast`, `normalised_rms_distance`) in single-snapshot files and capitalised (`Centroid_Distance`, `MRL`, `Pairwise`, `Hemi_Contrast`, `Normalised_RMS_Distance`) in evolution files. Consumers must check both (`analysis._extract_metric_value()`).

### 2. Pairwise > 90° Cap Rule

Systems with `Pairwise > 90.0°` represent anti-clustered configurations and are excluded when ranking for lopsidedness (`extract_ranks.py`, `extract_extremes.py`, `analysis.process_raw_overview_data()`).

### 3. Analysis-Layer DataFrame Contracts (`analysis.py`) `[PENDING: Human | AI-Assisted | AI-Developed]`

All functions in `analysis.py` obey the following rules:

- **No implicit column assumptions**: Required columns are validated via `_require_columns()` which raises `KeyError` with a descriptive message.
- **No silent in-place mutation**: Input DataFrames are copied before modification.
- **Explicit NaN handling**: Missing values are represented as `np.nan` and dropped/handled with documented rules.
- **Derived columns** introduced by `analysis.py`:
  - `lopsided_flag` (`"Top"` / `"Bottom"`)
  - `merger_class` (`"Stable"`, `"Infall"`, `"Fly-by"`, `"Loss"`)
  - `time_since_last_major_merger` (int, snapshots)
  - `num_mergers` (int)
  - `mean_mag_gap` (float)
  - `var_mag_gap` (float)
  - `merger_time_fraction` (float in `[0, 1]`)

---

## Statistical Tail Conventions (`p_value_engine.py`) `[PENDING: Human | AI-Assisted | AI-Developed]`

| Metric | Key | Tail Direction | Extreme Direction |
|:-------|:----|:---------------|:------------------|
| Mean Resultant Length | `MRL` / `R_bar` | **Right-tailed** ($p = P(\text{null} \geq \text{obs})$) | Larger = more lopsided |
| Pairwise Angle | `Pairwise` / `theta_lop` | **Left-tailed** ($p = P(\text{null} \leq \text{obs})$) | Smaller = more lopsided |
| Hemispheric Contrast | `Hemi_Contrast` / `C_hemi` | **Right-tailed** ($p = P(\text{null} \geq \text{obs})$) | Larger = more lopsided |
| Centroid Offset | `Centroid_Offset` / `D_cen` | **Right-tailed** ($p = P(\text{null} \geq \text{obs})$) | Larger = more lopsided |
