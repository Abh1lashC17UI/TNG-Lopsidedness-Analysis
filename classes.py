import illustris_python as il
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import pyvista as pv
import trimesh
from tqdm import tqdm
import os
import pickle
import hashlib

import metrics
import fetch

#DEBUGTHIS = True
DEBUGTHIS = False

# Cache wrapper for Halo methods
def cacheable_method(func):
    def wrapper(self, *args, force_rerun=False, **kwargs):
        cache_key = f"__cache_{func.__name__}"
        if not force_rerun and getattr(self, cache_key, False):
            if DEBUGTHIS:
                print(f"[cache] Skipping {func.__name__}() — already computed.")
            return
        result = func(self, *args, **kwargs)
        setattr(self, cache_key, True)
        self._autosave()
        return result
    return wrapper


class Halo:
    """
        Class for halo.
        Stores list of subhalo indices under its influence, computes metrics,
        analyzes Top 10 massive satellites, and parses merger tree history.
    """

    _cache_dir = ".halo_cache"
    _loaded_instances = {}

    def __new__(cls, halos=None, index=None, fields=None, LBox=None, Current_Snap=None, Sim_Name=None):
        if index is None:
            return super().__new__(cls)

        key = (Sim_Name, index, Current_Snap)
        path = cls._get_cache_path(key)

        if key in cls._loaded_instances:
            return cls._loaded_instances[key]

        if os.path.exists(path):
            with open(path, "rb") as f:
                instance = pickle.load(f)
                cls._loaded_instances[key] = instance
                return instance

        instance = super().__new__(cls)
        cls._loaded_instances[key] = instance
        return instance

    def __init__(self, halos=None, index=None, fields=None, LBox=None, Current_Snap=None, Sim_Name=None):
        # If loaded from cache via __new__, skip initialization
        if hasattr(self, 'initialized'):
            return
            
        # If cache wasn't found, we need the raw catalog to build it
        if halos is None or fields is None:
            raise ValueError(f"Cache not found for Halo {index} in {Sim_Name} Snap {Current_Snap}. You must provide 'halos' and 'fields' to initialize a new instance.")

        self.initialized = True

        self.index = index
        self.fields = fields
        self.LBox = LBox
        self.Current_Snap = Current_Snap
        self.Sim_Name = Sim_Name

        self.properties = {field: halos[field][index] for field in fields}
        self.position = self.properties['GroupCM'].astype('float64')
        
        self.true_subhalo_positions = []
        self.true_dark_subhalo_positions = []
        self.subhalo_positions = []
        self.dark_subhalo_positions = []

        self.metrics = {"Central_Galaxy_Mass":0., "Central_Galaxy_BH":0., "Central_Galaxy_Vel":0.}
        self.metrics.update(dict.fromkeys(["centroid", "mrl", "pairwise"]))
        
        if 'GroupFirstSub' in fields and 'GroupNsubs' in fields:
            self.subhalo_indices = list(range(halos['GroupFirstSub'][index],
                halos['GroupFirstSub'][index]+halos['GroupNsubs'][index]))
            self.metrics['subhalo_count'] = int(halos['GroupNsubs'][index])
        else:
            self.subhalo_indices = None

        self.visible_subhalo_indices = []
        self.visible_subhalos = []
        self.dark_subhalos = []
        self.visible_subhalos_through_snaps = []
        self.central_galaxy_evolution = {"Snaps":[], "Positions":[], "IDs_over_snaps":[]}


    def __del__(self):
        return

    @classmethod
    def _get_cache_path(cls, key):
        # Simplified smart caching: purely depends on Simulation, Halo ID, and Snapshot
        Sim_Name, index, snap = key
        filename = f"sim_{Sim_Name}_halo_{index}_snap_{snap}.pkl"

        os.makedirs(cls._cache_dir, exist_ok=True)
        return os.path.join(cls._cache_dir, filename)

    def _autosave(self):
        key = (self.Sim_Name, self.index, self.Current_Snap)
        path = self._get_cache_path(key)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @cacheable_method
    def get_subhalos(self, subhalos, subhalo_fields):
        self.true_subhalo_positions = []
        self.true_dark_subhalo_positions = []
        self.subhalo_positions = []
        self.dark_subhalo_positions = []
        self.visible_subhalo_indices = []
        self.visible_subhalos = []
        self.dark_subhalos = []

        for subhalo_index in self.subhalo_indices:
            subhalo = Subhalo(subhalos, subhalo_index, subhalo_fields)
            if subhalo.is_subhalo():
                if subhalo.star_metals():
                    self.visible_subhalos.append(subhalo)
                    self.visible_subhalo_indices.append(subhalo_index)
                else:
                    self.dark_subhalos.append(subhalo)
            else:
                del subhalo

        if len(self.visible_subhalos)>1:
            central_subhalo_cm = self.visible_subhalos[0].properties['SubhaloPos'].astype('float64')
            self.metrics["Central_Galaxy_Mass"] = self.visible_subhalos[0].properties['SubhaloMass'].astype('float64')
            self.metrics["Central_Galaxy_BH"] = self.visible_subhalos[0].properties['SubhaloBHMdot'].astype('float64')
            self.metrics["Central_Galaxy_Vel"] = self.visible_subhalos[0].properties['SubhaloVel'].astype('float64')
        else:
            central_subhalo_cm = np.array([0., 0., 0.])

        for subhalo in self.visible_subhalos:
            self.true_subhalo_positions.append(subhalo.properties['SubhaloPos'].astype('float64'))
            self.subhalo_positions.append(subhalo.properties['SubhaloPos'].astype('float64') - central_subhalo_cm)

        for subhalo in self.dark_subhalos:
            self.true_dark_subhalo_positions.append(subhalo.properties['SubhaloPos'].astype('float64'))
            self.dark_subhalo_positions.append(subhalo.properties['SubhaloPos'].astype('float64') - central_subhalo_cm)

        self.true_subhalo_positions, self.subhalo_positions = self.correct_border(self.true_subhalo_positions, self.subhalo_positions)
        self.true_dark_subhalo_positions, self.dark_subhalo_positions = self.correct_border(self.true_dark_subhalo_positions, self.dark_subhalo_positions)

        self.subhalo_positions = np.array(self.subhalo_positions)
        self.dark_subhalo_positions = np.array(self.dark_subhalo_positions)

        self.metrics['visible_subhalos'] = len(self.visible_subhalos)

        rms = np.sqrt(np.mean(np.sum(self.subhalo_positions[1:]**2, axis=1)))
        rms_normalised = rms/self.properties['Group_R_Crit200']
        self.metrics['normalised_rms_distance'] = rms_normalised
        return

    @cacheable_method
    def get_subhalos_by_volume(self, subhalos, subhalo_fields, central_subhalo_index, subhalo_pos):
        self.true_subhalo_positions = []
        self.true_dark_subhalo_positions = []
        self.subhalo_positions = []
        self.dark_subhalo_positions = []
        self.visible_subhalo_indices = []
        self.visible_subhalos = []
        self.dark_subhalos = []

        central_subhalo = Subhalo(subhalos, central_subhalo_index, subhalo_fields)
        if not (central_subhalo.is_subhalo() and central_subhalo.star_metals()):
            pass

        central_subhalo_cm = central_subhalo.properties['SubhaloPos'].astype('float64')
        self.metrics["Central_Galaxy_Mass"] = central_subhalo.properties['SubhaloMass'].astype('float64')
        self.metrics["Central_Galaxy_BH"] = central_subhalo.properties['SubhaloBHMdot'].astype('float64')
        self.metrics["Central_Galaxy_Vel"] = central_subhalo.properties['SubhaloVel'].astype('float64')
        self.visible_subhalos.append(central_subhalo)
        self.visible_subhalo_indices.append(central_subhalo_index)
        self.subhalo_indices.append(central_subhalo_index)

        self.true_subhalo_positions.append(central_subhalo_cm)
        self.subhalo_positions.append(np.array([0., 0., 0.]))
        del central_subhalo

        self.subhalo_indices = []
        subhalo_pos = np.array(subhalo_pos)

        delta = subhalo_pos - central_subhalo_cm
        LBox_half = self.LBox / 2.0
        delta[delta > LBox_half] -= self.LBox
        delta[delta < -LBox_half] += self.LBox

        squared_distance_pbc = np.sum(delta**2, axis=1)
        distance_from_central_pbc = np.sqrt(squared_distance_pbc)

        distance_based_mask = distance_from_central_pbc <= 2 * self.properties['Group_R_Crit200']
        original_indices = np.where(distance_based_mask)[0]

        for subhalo_index in original_indices:
            if subhalo_index == central_subhalo_index:
                continue

            subhalo = Subhalo(subhalos, subhalo_index, subhalo_fields)
            self.subhalo_indices.append(subhalo_index)

            if subhalo.is_subhalo:
                if subhalo.star_metals():
                    self.visible_subhalos.append(subhalo)
                    self.visible_subhalo_indices.append(subhalo_index)
                else:
                    self.dark_subhalos.append(subhalo)
            else:
                del subhalo

        if len(self.visible_subhalos)==1:
            central_subhalo_cm = np.array([0., 0., 0.])

        for subhalo in self.visible_subhalos:
            subhalo_original_index = subhalo.index
            if subhalo_original_index == central_subhalo_index:
                continue
            idx_in_original_indices = np.where(original_indices == subhalo_original_index)[0][0]
            corrected_delta = delta[original_indices][idx_in_original_indices]

            self.true_subhalo_positions.append(subhalo.properties['SubhaloPos'].astype('float64'))
            self.subhalo_positions.append(corrected_delta)

        for subhalo in self.dark_subhalos:
            subhalo_original_index = subhalo.index
            idx_in_original_indices = np.where(original_indices == subhalo_original_index)[0][0]
            corrected_delta = delta[original_indices][idx_in_original_indices]

            self.true_dark_subhalo_positions.append(subhalo.properties['SubhaloPos'].astype('float64'))
            self.dark_subhalo_positions.append(corrected_delta)

        self.subhalo_positions = np.array(self.subhalo_positions)
        self.dark_subhalo_positions = np.array(self.dark_subhalo_positions)
        self.metrics['visible_subhalos'] = len(self.visible_subhalos)

        if self.subhalo_positions.size > 0:
            rms = np.sqrt(np.mean(np.sum(self.subhalo_positions[1:]**2, axis=1)))
            rms_normalised = rms / self.properties['Group_R_Crit200']
            self.metrics['normalised_rms_distance'] = rms_normalised
        else:
            self.metrics['normalised_rms_distance'] = 0.0

    @cacheable_method
    def calc_metrics(self):
        self.metrics['centroid'] = metrics.Centroid.eval(self.subhalo_positions[1:])
        self.metrics['mrl'] = metrics.MRL.eval(self.subhalo_positions[1:])
        self.metrics['pairwise'] = metrics.Pairwise.eval(self.subhalo_positions[1:])
        self.metrics['hemi_contrast'], self.metrics['hemi'], self.metrics['hemi_vector'] = metrics.Hemi.eval(self.subhalo_positions[1:], num_grid=30)

        if [0., 0., 0.] in self.dark_subhalo_positions[0]:
            self.dark_subhalo_positions = self.dark_subhalo_positions[1:]
            self.dark_subhalos = self.dark_subhalos[1:]

        all_subhalo_positions = np.array(list(self.subhalo_positions[1:]) + list(self.dark_subhalo_positions))
        self.metrics['dark_centroid'] = metrics.Centroid.eval(all_subhalo_positions)
        self.metrics['dark_mrl'] = metrics.MRL.eval(all_subhalo_positions)
        self.metrics['dark_pairwise'] = metrics.Pairwise.eval(all_subhalo_positions)
        self.metrics['dark_hemi_contrast'], self.metrics['dark_hemi'], self.metrics['dark_hemi_vector'] = metrics.Hemi.eval(all_subhalo_positions, num_grid=30)
        return

    @cacheable_method
    def track_visibile_subhalos(self, stop_after=0):
        self.central_galaxy_evolution = {"Snaps":[], "Positions":[], "IDs_over_snaps":[]}
        self.visible_subhalos_through_snaps = []

        for i, id in tqdm(enumerate(self.visible_subhalo_indices), total=len(self.visible_subhalo_indices), desc="Processing Subhalo Tracks"):
            time_evolution = {"ID_in_current_snap": id, "Past_Positons":[], "Seen_in_snaps":None, "IDs_over_snaps": None}

            try:
                pos, snapnum, subID = fetch.get_subhalo_ids(id, self.Current_Snap, self.LBox, sim_name = self.Sim_Name)

                if i==0:
                    self.central_galaxy_evolution["Positions"] = pos
                    self.central_galaxy_evolution["Snaps"] = snapnum
                    self.central_galaxy_evolution["IDs_over_snaps"] = subID
                    time_evolution["Past_Positons"] = pos
                    trimmed_snaps = snapnum

                if i > 0:
                    cg_indices, pos_indices = [], []
                    for j in range(len(snapnum)):
                        for k in range(len(self.central_galaxy_evolution["Snaps"])):
                            if snapnum[j] == self.central_galaxy_evolution["Snaps"][k]:
                                pos_indices.append(j)
                                cg_indices.append(k)

                    pos[pos_indices] -= self.central_galaxy_evolution["Positions"][cg_indices]
                    pos = pos[pos_indices]
                    snapnum = snapnum[pos_indices]

                    current_index = np.where(snapnum == self.Current_Snap)[0][0]
                    start_index = current_index - 31 if current_index > 31 else 0

                    time_evolution["Past_Positons"] = pos[start_index : current_index + 1]
                    trimmed_snaps = snapnum[start_index : current_index + 1]

            except Exception as e:
                pos, snapnum, subID = [], [], []
                trimmed_snaps = []

            time_evolution["Seen_in_snaps"] = trimmed_snaps
            time_evolution["IDs_over_snaps"] = subID
            self.visible_subhalos_through_snaps.append(time_evolution)

        return

    @cacheable_method
    def calc_metrics_over_time_using_tracks(self):
        self.satellite_metrics_over_time = {
            "Snaps": [], "Centroid_Distance": [], "MRL": [],
            "Pairwise": [], "Hemi_Contrast": [], "Normalised_RMS_Distance": []
        }

        all_snaps = set()
        for evolution in self.visible_subhalos_through_snaps[1:]: 
            if evolution["Seen_in_snaps"] is not None and len(evolution["Seen_in_snaps"]) > 0:
                all_snaps.update(evolution["Seen_in_snaps"].tolist()) 

        sorted_snaps = sorted(list(all_snaps))
        satellite_tracks = self.visible_subhalos_through_snaps[1:]

        for snap in tqdm(sorted_snaps, desc="Calculating Metrics Over Time"):
            satellite_positions_at_snap = []

            for satellite_evo in satellite_tracks:
                if satellite_evo["Seen_in_snaps"] is None:
                    continue
                snap_indices = np.where(satellite_evo["Seen_in_snaps"] == snap)[0]
                if snap_indices.size > 0:
                    sat_snap_index = snap_indices[0]
                    sat_pos_at_snap = satellite_evo["Past_Positons"][sat_snap_index]
                    satellite_positions_at_snap.append(sat_pos_at_snap)

            if len(satellite_positions_at_snap) > 1:
                sat_pos_array = np.array(satellite_positions_at_snap, dtype=np.float64)

                self.satellite_metrics_over_time["Snaps"].append(snap)
                
                c_dist = metrics.Centroid.eval(sat_pos_array)
                self.satellite_metrics_over_time["Centroid_Distance"].append(c_dist if isinstance(c_dist, float) else np.linalg.norm(c_dist))
                self.satellite_metrics_over_time["MRL"].append(metrics.MRL.eval(sat_pos_array))
                self.satellite_metrics_over_time["Pairwise"].append(metrics.Pairwise.eval(sat_pos_array))
                hemi_contrast, _, _ = metrics.Hemi.eval(sat_pos_array, num_grid=10)
                self.satellite_metrics_over_time["Hemi_Contrast"].append(hemi_contrast)

                dists_sq = np.sum(sat_pos_array**2, axis=1)
                d_rms = np.sqrt(np.mean(dists_sq))
                self.satellite_metrics_over_time["Normalised_RMS_Distance"].append(d_rms / self.properties['Group_R_Crit200'])

        return

    @cacheable_method
    def process_top_10(self):
        """ Isolates the 10 most massive satellites and computes their specific metrics. """
        if len(self.visible_subhalos) <= 1:
            self.top_10_relative_indices = []
            self.top_10_metrics = {}
            return

        # Associate relative indices with satellites (skip index 0, which is Central)
        sats_with_idx = [(i, self.visible_subhalos[i]) for i in range(1, len(self.visible_subhalos))]
        sorted_sats_with_idx = sorted(sats_with_idx, key=lambda x: x[1].properties['SubhaloMass'], reverse=True)[:10]

        self.top_10_relative_indices = [x[0] for x in sorted_sats_with_idx]
        
        # Calculate isolated metrics
        top_10_pos_array = np.array([self.subhalo_positions[i] for i in self.top_10_relative_indices])
        if len(top_10_pos_array) > 0:
            c_dist = metrics.Centroid.eval(top_10_pos_array)
            self.top_10_metrics = {
                'centroid': c_dist if isinstance(c_dist, float) else np.linalg.norm(c_dist),
                'mrl': metrics.MRL.eval(top_10_pos_array),
                'pairwise': metrics.Pairwise.eval(top_10_pos_array) if len(top_10_pos_array) > 1 else np.nan,
                'hemi_contrast': metrics.Hemi.eval(top_10_pos_array, num_grid=30)[0],
                'normalised_rms_distance': np.sqrt(np.mean(np.sum(top_10_pos_array**2, axis=1))) / self.properties['Group_R_Crit200'],
                'subhalo_count': len(top_10_pos_array)
            }
        else:
            self.top_10_metrics = {}

    @cacheable_method
    def calc_top_10_metrics_over_time(self):
        """ Calculates the time evolution of metrics solely for the Top 10 tracked satellites. """
        if not hasattr(self, 'top_10_relative_indices'):
            self.process_top_10()

        self.top_10_metrics_over_time = {
            "Snaps": [], "Centroid_Distance": [], "MRL": [],
            "Pairwise": [], "Hemi_Contrast": [], "Normalised_RMS_Distance": []
        }

        if not self.top_10_relative_indices:
            return

        all_snaps = set()
        for idx in self.top_10_relative_indices:
            evolution = self.visible_subhalos_through_snaps[idx]
            if evolution["Seen_in_snaps"] is not None and len(evolution["Seen_in_snaps"]) > 0:
                all_snaps.update(evolution["Seen_in_snaps"].tolist())

        sorted_snaps = sorted(list(all_snaps))

        for snap in sorted_snaps:
            step_pos = []
            for idx in self.top_10_relative_indices:
                evo = self.visible_subhalos_through_snaps[idx]
                if evo["Seen_in_snaps"] is None: continue
                snap_idx = np.where(evo["Seen_in_snaps"] == snap)[0]
                if snap_idx.size > 0:
                    step_pos.append(evo["Past_Positons"][snap_idx[0]])

            if len(step_pos) > 1:
                pos_arr = np.array(step_pos, dtype=np.float64)
                self.top_10_metrics_over_time["Snaps"].append(snap)
                
                c_dist = metrics.Centroid.eval(pos_arr)
                self.top_10_metrics_over_time["Centroid_Distance"].append(c_dist if isinstance(c_dist, float) else np.linalg.norm(c_dist))
                self.top_10_metrics_over_time["MRL"].append(metrics.MRL.eval(pos_arr))
                self.top_10_metrics_over_time["Pairwise"].append(metrics.Pairwise.eval(pos_arr))
                self.top_10_metrics_over_time["Hemi_Contrast"].append(metrics.Hemi.eval(pos_arr, num_grid=10)[0])
                self.top_10_metrics_over_time["Normalised_RMS_Distance"].append(np.sqrt(np.mean(np.sum(pos_arr**2, axis=1))) / self.properties['Group_R_Crit200'])

    def _are_related(self, basePath, ancestor_id, ancestor_snap, descendant_id, descendant_snap):
        """ Helper for Merger Analysis to check lineage continuity between two subhalos. """
        try:
            tree = il.sublink.loadTree(basePath, descendant_snap, descendant_id, fields=['SnapNum', 'SubfindID'], onlyMPB=True)
            if tree is None: return False
            match_idx = np.where(tree['SnapNum'] == ancestor_snap)[0]
            if len(match_idx) > 0:
                return tree['SubfindID'][match_idx[0]] == ancestor_id
        except Exception:
            pass
        return False

    @cacheable_method
    def analyze_merger_history(self, basePath, min_snap=68, band_index=5):
        """ Classifies a halo's recent interaction history. """
        if not self.visible_subhalo_indices:
            self.merger_class = "Unknown"
            self.merger_history = []
            return

        central_sub_id = self.visible_subhalo_indices[0]
        tree = il.sublink.loadTree(basePath, self.Current_Snap, central_sub_id, fields=['SnapNum', 'SubfindID', 'SubhaloGrNr'], onlyMPB=True)
        
        if tree is None or 'SnapNum' not in tree:
            self.merger_class = "Unknown"
            self.merger_history = []
            return

        history = []
        for i in range(len(tree['SnapNum'])):
            snap = tree['SnapNum'][i]
            if snap < min_snap or snap > self.Current_Snap:
                continue

            c_id = tree['SubfindID'][i]
            g_id = tree['SubhaloGrNr'][i]

            try:
                group_cat = il.groupcat.loadHalos(basePath, snap, fields=['GroupFirstSub', 'GroupNsubs'])
                first_sub = group_cat['GroupFirstSub'][g_id]
                n_subs = group_cat['GroupNsubs'][g_id]

                if n_subs <= 1: continue

                local_central_idx = c_id - first_sub
                if local_central_idx < 0 or local_central_idx >= n_subs: continue

                sub_mags_all = il.groupcat.loadSubhalos(basePath, snap, fields=['SubhaloStellarPhotometrics'])
                if isinstance(sub_mags_all, dict):
                    if 'SubhaloStellarPhotometrics' in sub_mags_all:
                        sub_mags_all = sub_mags_all['SubhaloStellarPhotometrics']
                    elif 'SubhaloPhotometrics' in sub_mags_all:
                        sub_mags_all = sub_mags_all['SubhaloPhotometrics']

                group_mags = sub_mags_all[first_sub : first_sub + n_subs, band_index]
                central_mag = group_mags[local_central_idx]

                mask = np.ones(n_subs, dtype=bool)
                mask[local_central_idx] = False

                sat_mags = group_mags[mask]
                if len(sat_mags) == 0: continue

                brightest_idx_local = np.argmin(sat_mags)
                sat_mag = sat_mags[brightest_idx_local]
                sat_id = np.arange(first_sub, first_sub + n_subs)[mask][brightest_idx_local]

                history.append({
                    'SnapNum': snap,
                    'Mag_Central': central_mag,
                    'Brightest_Sat_Mag': sat_mag,
                    'Sat_ID': sat_id,
                    'Mag_Gap': sat_mag - central_mag
                })
            except Exception:
                pass

        self.merger_history = history

        # Classification Logic
        df_group = pd.DataFrame(history)
        if df_group.empty or len(df_group) < 3:
            self.merger_class = "Stable"
            return

        df_group = df_group.sort_values('SnapNum')
        gaps = df_group['Mag_Gap'].values
        ids = df_group['Sat_ID'].values
        snaps = df_group['SnapNum'].values

        min_gap = np.min(gaps)
        if (gaps[-1] - gaps[0]) > 1.5:
            self.merger_class = "Stripping"
        elif min_gap < 1.5:
            min_idx = np.argmin(gaps)
            sat_id_at_min = ids[min_idx]
            snap_at_min = snaps[min_idx]

            future_gaps = gaps[min_idx:]
            future_ids = ids[min_idx:]
            future_snaps = snaps[min_idx:]

            if len(future_gaps) > 1:
                max_future_gap = np.max(future_gaps)
                if max_future_gap > 1.5:
                    recovery_indices = np.where(future_gaps > 1.5)[0]
                    if len(recovery_indices) > 0:
                        first_rec = recovery_indices[0]
                        if self._are_related(basePath, sat_id_at_min, snap_at_min, future_ids[first_rec], future_snaps[first_rec]):
                            self.merger_class = "Fly-by"
                        else:
                            self.merger_class = "Infall"
                else:
                    self.merger_class = "Infall"
            else:
                self.merger_class = "Infall"
        else:
            self.merger_class = "Stable"


    def correct_tracks(self):
        for i  in range(1, len(self.visible_subhalo_indices)):
            pos = self.visible_subhalos_through_snaps[i]["Past_Positons"]
            if len(pos) == 0: continue
            diff = pos[-1] - self.subhalo_positions[i]
            if len(diff[diff > self.LBox/2])>0 or len(diff[diff < -self.LBox / 2])>0:
                temp_pos = pos
                temp_pos[:, diff > self.LBox/2] -= self.LBox
                temp_pos[:, diff < -self.LBox/2] += self.LBox
                pos = temp_pos
            self.visible_subhalos_through_snaps[i]["Past_Positons"] = pos

    def plot(self, no_tracks=False):
        fig = plt.figure()
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_3d.scatter(0, 0, 0, color = 'red', label="Central Subhalo", s=5)
        ax_3d.scatter(self.subhalo_positions[1:, 0], self.subhalo_positions[1:, 1], self.subhalo_positions[1:, 2], color = 'green', label="Visible Subhalos", s=3)
        ax_3d.scatter(self.dark_subhalo_positions[1:, 0], self.dark_subhalo_positions[1:, 1], self.dark_subhalo_positions[1:, 2], color = 'black', label="Dark Subhalos", s=1)

        if no_tracks:
            self.visible_subhalos_through_snaps = []

        for i, visible_subhalo in enumerate(self.visible_subhalos_through_snaps):
            past_positions = np.array(visible_subhalo["Past_Positons"])
            if i>0 and len(past_positions)>0:
                ax_3d.plot(past_positions[:, 0], past_positions[:, 1], past_positions[:, 2], color='goldenrod', linewidth=1)
            del past_positions

        custom_lines = [Line2D([0], [0], color='goldenrod', lw=1)]
        custom_labels = ['Past Track']
        ax_3d.set_title("Subhalos bound to the halo with tracks of the visible subhalos")
        ax_3d.legend(handles=ax_3d.get_legend_handles_labels()[0] + custom_lines, labels=ax_3d.get_legend_handles_labels()[1] + custom_labels)
        plt.show()

    def correct_border(self, true_pos, pos):
        if len(true_pos) == 0: return true_pos, pos
        raw_diff  = true_pos - true_pos[0]
        raw_diff[raw_diff > self.LBox / 2] -= self.LBox
        raw_diff[raw_diff < -self.LBox / 2] += self.LBox
        true_pos[:] = raw_diff + true_pos[0]
        pos[:] = true_pos - true_pos[0]
        return true_pos, pos

    def save(self, save_metrics=True, save_subhalo_pos=True, save_dark_subhalo=True, save_tracks=True, 
             save_metric_evolution_using_tracks=True, save_central_ids=True, 
             save_top_10=True, save_merger_analysis=True, outdir=""):
             
        base_dir = "halo_exports"
        final_outdir = os.path.join(base_dir, outdir) if outdir else base_dir
        os.makedirs(final_outdir, exist_ok=True)

        if save_metrics:
            metrics_path = os.path.join(final_outdir, f"sim_{self.Sim_Name}_halo_{self.index}_snap_{self.Current_Snap}_metrics.csv")
            metrics_df = pd.DataFrame([self.properties | self.metrics])
            metrics_df.to_csv(metrics_path, index=False)

        if save_subhalo_pos:
            filename = f"sim_{self.Sim_Name}_halo_{self.index}_snap_{self.Current_Snap}_positions.csv"
            for i, subhalo in enumerate(self.visible_subhalos[1:], start=1):
                subhalo.save(final_outdir, filename, relative_pos=self.subhalo_positions[i], mode='a')

        if save_dark_subhalo:
            filename = f"sim_{self.Sim_Name}_halo_{self.index}_snap_{self.Current_Snap}_dark_positions.csv"
            for i, subhalo in enumerate(self.dark_subhalos):
                subhalo.save(final_outdir, filename, relative_pos=self.dark_subhalo_positions[i], mode='a')

        if save_tracks:
            all_tracks = []
            for i, subhalo in enumerate(self.visible_subhalos_through_snaps):
                if i==0: continue
                track = subhalo["Past_Positons"]
                if track is not None and len(track) > 0:
                    df = pd.DataFrame(track, columns=["x", "y", "z"])
                    df["subhalo_index"] = i
                    all_tracks.append(df)
            if all_tracks:
                pd.concat(all_tracks, ignore_index=True).to_csv(os.path.join(final_outdir, f"sim_{self.Sim_Name}_halo_{self.index}_snap_{self.Current_Snap}_all_tracks.csv"), index=False)

        if save_metric_evolution_using_tracks:
            if not hasattr(self, 'satellite_metrics_over_time'):
                self.calc_metrics_over_time_using_tracks()
            if hasattr(self, 'satellite_metrics_over_time') and self.satellite_metrics_over_time["Snaps"]:
                evolution_path = os.path.join(final_outdir, f"sim_{self.Sim_Name}_halo_{self.index}_snap_{self.Current_Snap}_metric_evolution.csv")
                data = {
                    "SnapNum": self.satellite_metrics_over_time["Snaps"],
                    "Centroid_Distance": [np.linalg.norm(c) if isinstance(c, np.ndarray) else c for c in self.satellite_metrics_over_time["Centroid_Distance"]],
                    "MRL": self.satellite_metrics_over_time["MRL"],
                    "Pairwise": self.satellite_metrics_over_time["Pairwise"],
                    "Hemi_Contrast": self.satellite_metrics_over_time["Hemi_Contrast"],
                    "Normalised_RMS_Distance": self.satellite_metrics_over_time['Normalised_RMS_Distance']
                }
                pd.DataFrame(data).to_csv(evolution_path, index=False)

        if save_central_ids:
            self.save_central_ids(final_outdir, f"sim_{self.Sim_Name}_halo_{self.index}_snap_{self.Current_Snap}_central_ids.csv")

        if save_top_10:
            if not hasattr(self, 'top_10_relative_indices'):
                self.process_top_10()
            if not hasattr(self, 'top_10_metrics_over_time'):
                self.calc_top_10_metrics_over_time()

            # Save Top 10 Positions
            pos_filename = f"sim_{self.Sim_Name}_halo_{self.index}_snap_{self.Current_Snap}_positions_top_10.csv"
            if len(self.visible_subhalos) > 0:
                self.visible_subhalos[0].save(final_outdir, pos_filename, relative_pos=np.array([0.,0.,0.]), mode='w')
            if hasattr(self, 'top_10_relative_indices'):
                for idx in self.top_10_relative_indices:
                    self.visible_subhalos[idx].save(final_outdir, pos_filename, relative_pos=self.subhalo_positions[idx], mode='a')

            # Save Top 10 Tracks
            all_tracks_top10 = []
            if hasattr(self, 'top_10_relative_indices'):
                for idx in self.top_10_relative_indices:
                    subhalo = self.visible_subhalos_through_snaps[idx]
                    track = subhalo["Past_Positons"]
                    if track is not None and len(track) > 0:
                        df_track = pd.DataFrame(track, columns=["x", "y", "z"])
                        df_track["subhalo_index"] = idx
                        all_tracks_top10.append(df_track)
            if all_tracks_top10:
                pd.concat(all_tracks_top10, ignore_index=True).to_csv(os.path.join(final_outdir, f"sim_{self.Sim_Name}_halo_{self.index}_snap_{self.Current_Snap}_all_tracks_top_10.csv"), index=False)

            # Save Top 10 Metrics
            if hasattr(self, 'top_10_metrics') and self.top_10_metrics:
                top_10_out = self.top_10_metrics.copy()
                top_10_out['HaloID'] = self.index
                top_10_out['SnapNum'] = self.Current_Snap
                pd.DataFrame([top_10_out]).to_csv(os.path.join(final_outdir, f"sim_{self.Sim_Name}_halo_{self.index}_snap_{self.Current_Snap}_metrics_top_10.csv"), index=False)

            # Save Top 10 Evolution
            if hasattr(self, 'top_10_metrics_over_time') and self.top_10_metrics_over_time["Snaps"]:
                data = {
                    "SnapNum": self.top_10_metrics_over_time["Snaps"],
                    "Centroid_Distance": [np.linalg.norm(c) if isinstance(c, np.ndarray) else c for c in self.top_10_metrics_over_time["Centroid_Distance"]],
                    "MRL": self.top_10_metrics_over_time["MRL"],
                    "Pairwise": self.top_10_metrics_over_time["Pairwise"],
                    "Hemi_Contrast": self.top_10_metrics_over_time["Hemi_Contrast"],
                    "Normalised_RMS_Distance": self.top_10_metrics_over_time['Normalised_RMS_Distance']
                }
                pd.DataFrame(data).to_csv(os.path.join(final_outdir, f"sim_{self.Sim_Name}_halo_{self.index}_snap_{self.Current_Snap}_metric_evolution_top_10.csv"), index=False)

        if save_merger_analysis:
            if hasattr(self, 'merger_history') and self.merger_history:
                df_hist = pd.DataFrame(self.merger_history)
                df_hist.insert(0, 'HaloID_Snap99', self.index)
                df_hist['Merger_Class'] = getattr(self, 'merger_class', 'Unknown')
                df_hist.to_csv(os.path.join(final_outdir, f"sim_{self.Sim_Name}_halo_{self.index}_snap_{self.Current_Snap}_merger_analysis.csv"), index=False)

        if DEBUGTHIS:
            print(f"Halo {self.index} data saved to '{final_outdir}/'")

    def save_central_ids(self, final_outdir, filename):
        if self.central_galaxy_evolution["Snaps"] is None or len(self.central_galaxy_evolution["Snaps"]) == 0:
            return

        data = {
            "SnapNum": self.central_galaxy_evolution["Snaps"],
            "SubfindID": np.flip(self.central_galaxy_evolution["IDs_over_snaps"], axis=0)
        }
        data["BoxSize"] = [self.LBox] * len(data["SnapNum"])
        data["CentralPos_x"] = [self.true_subhalo_positions[0][0]] * len(data["SnapNum"])
        data["CentralPos_y"] = [self.true_subhalo_positions[0][1]] * len(data["SnapNum"])
        data["CentralPos_z"] = [self.true_subhalo_positions[0][2]] * len(data["SnapNum"])

        pd.DataFrame(data).to_csv(os.path.join(final_outdir, filename), index=False)


class Subhalo:
    """ Class for subhalo. """

    def __init__(self, subhalos, index: int, fields: list):
        self.index = index
        self.fields = fields
        self.properties = {field: subhalos[field][index] for field in fields}

    def __del__(self):
        return

    def visible(self):
        return np.mean(self.properties['SubhaloStellarPhotometrics']) < -5

    def star_metals(self):
        return not (self.properties['SubhaloStarMetalFractions'][-1] == 0.)

    def is_subhalo(self):
        return self.properties['SubhaloFlag'] == 1

    def stellar_visibility(self):
        pass

    def stellarmass(self):
        return self.properties['SubhaloMassType'][4] * 1e10

    def save(self, outdir, filename, relative_pos=None, mode='a'):
        os.makedirs(outdir, exist_ok=True)
        pos = self.properties['SubhaloCM'].astype(float) if relative_pos is None else np.asarray(relative_pos, dtype=float)

        df = pd.DataFrame([{
            "x": pos[0], "y": pos[1], "z": pos[2],
            "subhalo_index": self.index, **self.properties
        }])

        file_path = os.path.join(outdir, filename)
        if not os.path.exists(file_path) or mode == 'w':
            df.to_csv(file_path, index=False, mode='w')
        else:
            df.to_csv(file_path, index=False, mode='a', header=False)