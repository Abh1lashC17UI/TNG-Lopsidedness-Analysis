"""
p_value_engine.py — Statistical Significance Engine
=====================================================
Computes p_iso (isotropic null model) and p_cosmo (cosmological ensemble)
for the four core lopsidedness metrics.

Architecture notes
------------------
- No UI, no plotting, no PyQt5.
- All cached null distributions live in .p_val_cache/ (never .halo_cache/).
- Pure functions take explicit inputs and return structured dicts.
- Metric tail conventions:
    MRL           → right-tailed  (higher = more lopsided)
    Hemi_Contrast → right-tailed  (higher = more lopsided)
    Centroid_Offset → right-tailed (higher = more lopsided)
    Pairwise      → left-tailed   (lower angle = more lopsided)
"""

import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
_tqdm = tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of Monte-Carlo iterations used when generating a null distribution.
N_ITER_DEFAULT: int = 100_000

#: Number of hemisphere-grid test directions used when computing C_hemi.
N_GRID_HEMI: int = 1_024

#: Batch size for vectorised hemisphere computation (controls peak memory use).
HEMI_BATCH: int = 2_000

#: Name of the cache pickle file inside .p_val_cache/.
CACHE_FILENAME: str = "isotropic_null_distributions.pkl"

#: Metrics and their tail directions.  True = right-tailed, False = left-tailed.
METRIC_RIGHT_TAIL: dict = {
    "MRL": True,
    "Hemi_Contrast": True,
    "Centroid_Offset": True,
    "Pairwise": False,   # lower angle → more clustered → more extreme
}

# In-process memo for cosmological ensemble DataFrames.
# Key: (base_export_folder, snap_num) → pd.DataFrame
_COSMO_ENSEMBLE_MEMO: dict = {}


# ---------------------------------------------------------------------------
# Helper: random unit vectors on S²
# ---------------------------------------------------------------------------

def _random_unit_vectors(n_systems: int, n_sat: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate *n_systems* independent sets of *n_sat* random unit vectors on S².

    Parameters
    ----------
    n_systems : int
        Number of independent realisations.
    n_sat : int
        Number of satellites per realisation.
    rng : np.random.Generator
        NumPy random generator (for reproducible seeding when desired).

    Returns
    -------
    np.ndarray
        Shape ``(n_systems, n_sat, 3)``, dtype float32.
    """
    # Rejection-free method: phi in [0, 2π), cos(theta) uniform in [-1, 1]
    cos_theta = rng.uniform(-1.0, 1.0, size=(n_systems, n_sat)).astype(np.float32)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2, dtype=np.float32)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(n_systems, n_sat)).astype(np.float32)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    return np.stack([x, y, z], axis=-1)  # (n_systems, n_sat, 3)


# ---------------------------------------------------------------------------
# Vectorised metric kernels for null distribution generation
# ---------------------------------------------------------------------------

def _compute_r_bar(uvecs: np.ndarray) -> np.ndarray:
    """
    Mean Resultant Length for a batch of realisations.

    Parameters
    ----------
    uvecs : np.ndarray
        Shape ``(n_systems, n_sat, 3)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_systems,)``, dtype float32.
    """
    mean_vec = uvecs.mean(axis=1)                          # (n_systems, 3)
    return np.linalg.norm(mean_vec, axis=1).astype(np.float32)


def _compute_theta_lop(uvecs: np.ndarray) -> np.ndarray:
    """
    Mean pairwise angle (degrees) for a batch of realisations.

    Uses the upper-triangle trick: for each realisation,
    compute the (n_sat × n_sat) dot-product matrix, extract the
    upper triangle, clip to [-1, 1] for numerical safety, then arccos.

    Parameters
    ----------
    uvecs : np.ndarray
        Shape ``(n_systems, n_sat, 3)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_systems,)``, dtype float32, in degrees.
    """
    n_systems, n_sat, _ = uvecs.shape
    results = np.empty(n_systems, dtype=np.float32)

    # Batch dot products: (n_systems, n_sat, n_sat)
    dots = np.einsum("bik,bjk->bij", uvecs, uvecs)  # float32 in, float32 out if input is f32

    # Upper triangle indices (excluding diagonal)
    i_idx, j_idx = np.triu_indices(n_sat, k=1)

    # Extract upper-triangle values for all systems at once: (n_systems, n_pairs)
    pairs = dots[:, i_idx, j_idx]                      # (n_systems, n_pairs)
    np.clip(pairs, -1.0, 1.0, out=pairs)
    angles = np.arccos(pairs)                          # radians
    results[:] = np.degrees(angles.mean(axis=1))
    return results


def _compute_c_hemi(uvecs: np.ndarray, grid_vecs: np.ndarray) -> np.ndarray:
    """
    Hemispheric contrast C_hemi = (2 * N_max - N_sat) / N_sat for a batch.

    Scans *grid_vecs* hemisphere directions and picks the one containing the
    most satellites.  Computed in sub-batches to cap peak memory.

    Parameters
    ----------
    uvecs : np.ndarray
        Shape ``(n_systems, n_sat, 3)``.
    grid_vecs : np.ndarray
        Shape ``(K, 3)`` — pre-generated isotropic grid directions.

    Returns
    -------
    np.ndarray
        Shape ``(n_systems,)``, dtype float32, in [0, 1].
    """
    n_systems, n_sat, _ = uvecs.shape
    results = np.empty(n_systems, dtype=np.float32)

    for start in range(0, n_systems, HEMI_BATCH):
        end = min(start + HEMI_BATCH, n_systems)
        batch = uvecs[start:end]                       # (B, n_sat, 3)
        # dots[b, s, k] = batch[b, s, :] · grid_vecs[k, :]
        dots = np.tensordot(batch, grid_vecs, axes=[[2], [1]])  # (B, n_sat, K)
        in_hemi = (dots >= 0.0)                        # (B, n_sat, K) bool
        n_plus = in_hemi.sum(axis=1)                   # (B, K)
        n_max = n_plus.max(axis=1)                     # (B,)
        results[start:end] = ((2.0 * n_max - n_sat) / n_sat).astype(np.float32)

    return results


# ---------------------------------------------------------------------------
# IsotropicNullEngine
# ---------------------------------------------------------------------------

class IsotropicNullEngine:
    """
    Manages the ``.p_val_cache/`` directory and isotropic null distributions.

    The cache is a single pickle file containing a dict of the form::

        {
            N_sat: {
                "R_bar":     np.ndarray  # (n_iter,) float32, sorted ascending
                "theta_lop": np.ndarray  # (n_iter,) float32, sorted ascending
                "C_hemi":    np.ndarray  # (n_iter,) float32, sorted ascending
            }
        }

    Parameters
    ----------
    cache_dir : str
        Absolute path to the ``.p_val_cache`` directory.  Created
        automatically if it does not exist.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir: str = cache_dir
        self.cache_file: str = os.path.join(cache_dir, CACHE_FILENAME)
        os.makedirs(cache_dir, exist_ok=True)
        self._distributions: dict = self._load_cache()

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict:
        """Load the cache pickle from disk; return empty dict on any failure."""
        if not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, "rb") as fh:
                data = pickle.load(fh)
            if not isinstance(data, dict):
                return {}
            return data
        except Exception as exc:
            warnings.warn(
                f"p_val_cache: could not load '{self.cache_file}': {exc}. "
                "Starting with an empty cache.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {}

    def _save_cache(self) -> None:
        """Persist the in-memory cache dict to disk."""
        try:
            with open(self.cache_file, "wb") as fh:
                pickle.dump(self._distributions, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            warnings.warn(
                f"p_val_cache: could not save '{self.cache_file}': {exc}.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_generate(
        self,
        n_sat: int,
        n_iter: int = N_ITER_DEFAULT,
        force_regenerate: bool = False,
    ) -> dict:
        """
        Return the sorted null distribution arrays for *n_sat* satellites.

        Parameters
        ----------
        n_sat : int
            Number of luminous satellites in the system.
        n_iter : int
            Monte-Carlo iterations (only used on cache miss).
        force_regenerate : bool
            If True, discard any cached entry and regenerate.

        Returns
        -------
        dict
            Keys: ``"R_bar"``, ``"theta_lop"``, ``"C_hemi"`` — each a
            sorted float32 ndarray of length *n_iter*.
        Raises
        ------
        ValueError
            If *n_sat* < 2 (metrics are undefined).
        """
        if n_sat < 2:
            raise ValueError(
                f"Cannot generate null distribution for N_sat={n_sat} < 2."
            )

        if force_regenerate or n_sat not in self._distributions:
            new_dist = self._generate_null_distribution(n_sat, n_iter)
            self._distributions[n_sat] = new_dist
            self._save_cache()
            return new_dist, True   # (dist, was_generated)

        return self._distributions[n_sat], False  # (dist, was_generated)

    def clear_entry(self, n_sat: int) -> None:
        """Remove a single N_sat entry from the cache (for force-reseed)."""
        if n_sat in self._distributions:
            del self._distributions[n_sat]
            self._save_cache()

    def clear_cache(self) -> None:
        """Wipe the entire cache both in memory and on disk."""
        self._distributions = {}
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Monte-Carlo generation
    # ------------------------------------------------------------------

    def _generate_null_distribution(self, n_sat: int, n_iter: int) -> dict:
        """
        Vectorised Monte-Carlo generation of the isotropic null distribution
        for *n_sat* satellites.

        A 4-step CLI progress bar is displayed while generating so the user
        can see that work is happening.  The bar is printed to ``stderr`` and
        is **only** shown during generation — cache hits never reach this path.

        Parameters
        ----------
        n_sat : int
        n_iter : int

        Returns
        -------
        dict with keys ``"R_bar"``, ``"theta_lop"``, ``"C_hemi"`` — each a
        sorted ascending float32 ndarray of length *n_iter*.
        """
        print(f"[Null Engine] Generating isotropic null distribution (N_sat={n_sat}, N_iter={n_iter:,})...")
        bar = tqdm(
            total=4,
            desc=f"[Null Engine] Sampling isotropic null (N_sat={n_sat:>3d})",
            unit="step",
            ncols=72,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            file=None,   # defaults to sys.stderr
            leave=False,
        )

        with bar:
            rng = np.random.default_rng()

            # Pre-generate isotropic grid vectors for C_hemi hemisphere scan
            grid_rng = np.random.default_rng(seed=42)
            grid_vecs = _random_unit_vectors(1, N_GRID_HEMI, grid_rng)[0]  # (K, 3)

            uvecs = _random_unit_vectors(n_iter, n_sat, rng)  # (n_iter, n_sat, 3)
            bar.set_postfix_str("R_bar …")
            bar.update(1)

            r_bar = _compute_r_bar(uvecs)
            r_bar.sort()
            bar.set_postfix_str("Pairwise …")
            bar.update(1)

            theta_lop = _compute_theta_lop(uvecs)
            theta_lop.sort()
            bar.set_postfix_str("C_hemi …")
            bar.update(1)

            c_hemi = _compute_c_hemi(uvecs, grid_vecs)
            c_hemi.sort()
            bar.set_postfix_str("done")
            bar.update(1)

        return {
            "R_bar":     r_bar,
            "theta_lop": theta_lop,
            "C_hemi":    c_hemi,
        }


# ---------------------------------------------------------------------------
# D_cen null distribution (on-the-fly, conditioned on actual radii)
# ---------------------------------------------------------------------------

def _compute_d_cen_null(
    n_sat: int,
    radii_r200: np.ndarray,
    n_iter: int = N_ITER_DEFAULT,
) -> np.ndarray:
    """
    Compute the isotropic null distribution for the Centroid Offset (D_cen)
    conditioned on the halo's actual satellite radii.

    The centroid offset is defined as::

        D_cen = |mean(r_i * r_hat_i)| / RMS_distance

    where ``r_i`` are the fixed observed radii (in units of R_200c) and
    ``r_hat_i`` are randomised unit direction vectors.

    Parameters
    ----------
    n_sat : int
        Nominal satellite count (from the metrics CSV).  The function
        derives the actual satellite count from ``len(radii_r200)`` to
        guarantee that the random-vector array and the radius array have
        identical shapes.  A mismatch (e.g. positions CSV has fewer rows
        than ``visible_subhalos``) would otherwise cause a broadcast error.
    radii_r200 : np.ndarray
        Shape ``(n_radii,)`` — satellite distances in units of R_200c.
        ``n_radii`` is used as the effective satellite count.
    n_iter : int

    Returns
    -------
    np.ndarray
        Shape ``(n_iter,)`` float32, sorted ascending.
    """
    # Use the actual length of the radii array as the satellite count so that
    # the shapes of uvecs (n_iter, n_radii, 3) and r (1, n_radii, 1) always
    # match, regardless of what n_sat says.
    n_radii = len(radii_r200)
    if n_radii < 1:
        return np.zeros(n_iter, dtype=np.float32)

    rng = np.random.default_rng()
    uvecs = _random_unit_vectors(n_iter, n_radii, rng)        # (n_iter, n_radii, 3)
    r = radii_r200.astype(np.float32)[np.newaxis, :, np.newaxis]  # (1, n_radii, 1)
    rms = float(np.sqrt(np.mean(radii_r200 ** 2)))
    if rms == 0.0:
        return np.zeros(n_iter, dtype=np.float32)

    # Weighted position vectors: (n_iter, n_radii, 3)
    pos = uvecs * r                                           # shapes always match
    centroid = pos.mean(axis=1)                               # (n_iter, 3)
    d_cen = np.linalg.norm(centroid, axis=1).astype(np.float32) / rms
    d_cen.sort()
    return d_cen


# ---------------------------------------------------------------------------
# p_iso computation
# ---------------------------------------------------------------------------

def compute_p_iso(
    engine: IsotropicNullEngine,
    metric_values: dict,
    n_sat: int,
    radii_r200: np.ndarray | None = None,
    n_iter: int = N_ITER_DEFAULT,
) -> tuple[dict, dict]:
    """
    Compute p_iso for the four core lopsidedness metrics.

    Parameters
    ----------
    engine : IsotropicNullEngine
    metric_values : dict
        Keys expected: ``"MRL"``, ``"Pairwise"``, ``"Hemi_Contrast"``,
        ``"Centroid_Offset"``.
    n_sat : int
        Number of luminous satellites.
    radii_r200 : np.ndarray or None
        Shape ``(n_sat,)`` — if provided, D_cen is computed on-the-fly using
        these actual radii.  If None, D_cen p_iso is returned as NaN.
    n_iter : int

    Returns
    -------
    p_iso : dict
        ``{metric_name: float}`` — NaN if metric is undefined for this halo.
    was_generated : bool
        True if the null distribution was freshly generated (cache miss).
    """
    nan = float("nan")

    if n_sat < 2:
        p_iso = {m: nan for m in METRIC_RIGHT_TAIL}
        return p_iso, False

    # Directional metrics (cached by N_sat)
    dist, was_generated = engine.get_or_generate(n_sat, n_iter=n_iter)

    p_iso: dict = {}

    # --- R_bar / MRL (right-tailed) ---
    obs_mrl = metric_values.get("MRL", nan)
    if np.isnan(obs_mrl):
        p_iso["MRL"] = nan
    else:
        idx = np.searchsorted(dist["R_bar"], obs_mrl)
        n = len(dist["R_bar"])
        p_iso["MRL"] = float(n - idx) / n          # fraction >= observed

    # --- theta_lop / Pairwise (left-tailed) ---
    obs_pw = metric_values.get("Pairwise", nan)
    if np.isnan(obs_pw):
        p_iso["Pairwise"] = nan
    else:
        idx = np.searchsorted(dist["theta_lop"], obs_pw, side="right")
        n = len(dist["theta_lop"])
        p_iso["Pairwise"] = float(idx) / n         # fraction <= observed

    # --- C_hemi / Hemi_Contrast (right-tailed) ---
    obs_hemi = metric_values.get("Hemi_Contrast", nan)
    if np.isnan(obs_hemi):
        p_iso["Hemi_Contrast"] = nan
    else:
        idx = np.searchsorted(dist["C_hemi"], obs_hemi)
        n = len(dist["C_hemi"])
        p_iso["Hemi_Contrast"] = float(n - idx) / n

    # --- D_cen / Centroid_Offset (right-tailed, on-the-fly) ---
    obs_cen = metric_values.get("Centroid_Offset", nan)
    if np.isnan(obs_cen) or radii_r200 is None or len(radii_r200) < 2:
        p_iso["Centroid_Offset"] = nan
    else:
        d_cen_null = _compute_d_cen_null(n_sat, radii_r200, n_iter=n_iter)
        idx = np.searchsorted(d_cen_null, obs_cen)
        n = len(d_cen_null)
        p_iso["Centroid_Offset"] = float(n - idx) / n

    return p_iso, was_generated


# ---------------------------------------------------------------------------
# p_cosmo computation
# ---------------------------------------------------------------------------

def compute_p_cosmo(
    metric_values: dict,
    cosmo_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """
    Compute cosmological p-values from an ensemble of halos at the same snapshot.

    Parameters
    ----------
    metric_values : dict
        Observed values keyed by ``"MRL"``, ``"Pairwise"``, ``"Hemi_Contrast"``,
        ``"Centroid_Offset"``.
    cosmo_df : pd.DataFrame
        DataFrame with columns ``mrl``, ``pairwise``, ``hemi_contrast``,
        ``centroid`` (snake_case, as read from ``*_metrics.csv``).
        Must contain at least one row.

    Returns
    -------
    p_cosmo : dict
        ``{metric_name: float}`` — fraction of ensemble halos more extreme.
    rank_info : dict
        ``{metric_name: (rank, total)}`` — 1-based rank (lower = more extreme).
    """
    nan = float("nan")

    # Mapping: our canonical key → CSV column name
    column_map = {
        "MRL": "mrl",
        "Pairwise": "pairwise",
        "Hemi_Contrast": "hemi_contrast",
        "Centroid_Offset": "centroid",
    }

    p_cosmo: dict = {}
    rank_info: dict = {}

    for metric_key, col in column_map.items():
        obs = metric_values.get(metric_key, nan)
        if np.isnan(obs) or col not in cosmo_df.columns:
            p_cosmo[metric_key] = nan
            rank_info[metric_key] = (nan, len(cosmo_df))
            continue

        vals = cosmo_df[col].dropna().values.astype(float)
        if len(vals) == 0:
            p_cosmo[metric_key] = nan
            rank_info[metric_key] = (nan, 0)
            continue

        right_tail = METRIC_RIGHT_TAIL[metric_key]

        if right_tail:
            # p = fraction of ensemble with value >= observed
            p_val = float(np.sum(vals >= obs)) / len(vals)
            # Rank: how many halos are MORE extreme (higher value)?
            rank = int(np.sum(vals > obs)) + 1          # 1-based
        else:
            # Left-tailed (Pairwise): p = fraction <= observed
            p_val = float(np.sum(vals <= obs)) / len(vals)
            rank = int(np.sum(vals < obs)) + 1

        p_cosmo[metric_key] = p_val
        rank_info[metric_key] = (rank, len(vals))

    return p_cosmo, rank_info


# ---------------------------------------------------------------------------
# Cosmological ensemble builder (memoised)
# ---------------------------------------------------------------------------

def build_cosmo_ensemble(
    base_export_folder: str,
    snap_num: int,
    top10: bool = False,
) -> pd.DataFrame:
    """
    Build the cosmological ensemble DataFrame for a given snapshot.

    Scans ``base_export_folder/Snap{snap_num}/*/`` for all
    ``*_metrics.csv`` (or ``*_metrics_top_10.csv`` when *top10* is True),
    reads the first row of each, and concatenates them.
    Results are memoised in-process separately for top-10 and full modes.

    When *top10* is True the returned DataFrame is additionally filtered to
    **only include halos that had at least 10 luminous satellites**.  The
    relevant column is ``subhalo_count`` in the top-10 file (which records
    the actual luminous count, capped at 10).  Any row where
    ``subhalo_count < 10`` or is missing is dropped, so that the ensemble
    denominator and rank calculations are restricted to systems for which a
    genuine top-10 selection exists.

    Parameters
    ----------
    base_export_folder : str
    snap_num : int
    top10 : bool
        If True, read ``*_metrics_top_10.csv`` files and restrict the
        ensemble to halos with ``subhalo_count >= 10``.
        If False (default), use the full ``*_metrics.csv`` ensemble without
        additional satellite-count filtering.

    Returns
    -------
    pd.DataFrame
        Columns: ``mrl``, ``pairwise``, ``hemi_contrast``, ``centroid``
        (and others depending on file schema).
        Empty DataFrame if no matching files or no halos pass the filter.
    """
    key = (base_export_folder, snap_num, top10)
    if key in _COSMO_ENSEMBLE_MEMO:
        return _COSMO_ENSEMBLE_MEMO[key]

    snap_dir = os.path.join(base_export_folder, f"Snap{snap_num}")
    if not os.path.isdir(snap_dir):
        _COSMO_ENSEMBLE_MEMO[key] = pd.DataFrame()
        return pd.DataFrame()

    # Choose file pattern based on mode
    if top10:
        pattern = os.path.join(snap_dir, "**", f"*_snap_{snap_num}_metrics_top_10.csv")
    else:
        pattern = os.path.join(snap_dir, "**", f"*_snap_{snap_num}_metrics.csv")

    metric_files = [
        f for f in glob.glob(pattern, recursive=True)
        if not (top10 is False and "_top_10" in os.path.basename(f))
    ]

    rows = []
    for fpath in tqdm(
        metric_files,
        desc=f"[Cosmo Ensemble] Aggregating Snap {snap_num} (N={len(metric_files)})",
        unit="halo",
        leave=False,
    ):
        try:
            df_row = pd.read_csv(fpath).iloc[0].to_dict()
            rows.append(df_row)
        except Exception:
            continue

    if not rows:
        _COSMO_ENSEMBLE_MEMO[key] = pd.DataFrame()
        return pd.DataFrame()

    ensemble = pd.DataFrame(rows)

    # Normalise column names: 'Centroid_Distance' -> 'centroid'
    if "Centroid_Distance" in ensemble.columns and "centroid" not in ensemble.columns:
        ensemble.rename(columns={"Centroid_Distance": "centroid"}, inplace=True)

    # When top10=True, restrict to halos with >= 10 luminous satellites.
    # In the top-10 metrics files 'subhalo_count' holds the actual luminous
    # count (capped at 10); a value < 10 means fewer than 10 luminous
    # satellites existed and the halo must be excluded from the reference
    # ensemble so that rank and total reflect only valid top-10 systems.
    if top10:
        sat_col = next(
            (c for c in ("subhalo_count", "visible_subhalos", "GroupNsubs")
             if c in ensemble.columns),
            None,
        )
        if sat_col is not None:
            before = len(ensemble)
            ensemble = ensemble[
                pd.to_numeric(ensemble[sat_col], errors="coerce").fillna(0) >= 10
            ].copy()
            after = len(ensemble)
            if after == 0:
                warnings.warn(
                    f"build_cosmo_ensemble(top10=True, snap={snap_num}): "
                    f"all {before} halos had fewer than 10 luminous satellites; "
                    "returning empty ensemble.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    _COSMO_ENSEMBLE_MEMO[key] = ensemble
    return ensemble


def invalidate_cosmo_ensemble_cache() -> None:
    """Clear the in-process memo for cosmological ensembles."""
    _COSMO_ENSEMBLE_MEMO.clear()
