import os
import glob
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


INPUT_FOLDER_PATH = r"C:\Users\Abhilash\Documents\Thesis\halo_exports\TNG100-1"

# Import your local metrics module
try:
    import metrics
except ImportError:
    print("CRITICAL ERROR: Could not import 'metrics.py'.")
    print("Please ensure 'metrics.py' and 'calc.py' are in the same directory as this script.")
    exit(1)

def calc_rms_normalized(positions, r200):
    """
    Calculates RMS distance normalized by R200.
    """
    if len(positions) == 0: return 0.0
    sq_dist = np.sum(positions**2, axis=1)
    rms = np.sqrt(np.mean(sq_dist))
    if r200 > 0:
        return rms / r200
    return 0.0

def process_single_snap(pos_file):
    """
    Reads a position file, filters top 10 visible subhalos, 
    saves new CSVs, and calculates metrics using metrics.py.
    """
    
    base_dir = os.path.dirname(pos_file)
    filename = os.path.basename(pos_file)
    
    # regex to capture base name (e.g. sim_TNG50-1_halo_67_snap_98)
    match = re.match(r"(.*)_positions\.csv", filename)
    if not match: return
    
    base_name = match.group(1)
    
    try:
        df_pos = pd.read_csv(pos_file)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    # Check structure (Need Central + at least 1 subhalo)
    if df_pos.empty or len(df_pos) < 2: return

    # The 'subhalo_index' in all_tracks.csv corresponds to the ROW NUMBER (0-based) 
    # in this positions file. We must capture this BEFORE sorting/filtering.
    # Note: Row 0 is Central, Row 1 is Sat #1, etc.
    df_pos['relative_track_id'] = range(len(df_pos))

    central_row = df_pos.iloc[[0]]
    satellites = df_pos.iloc[1:].copy()
    
    if len(satellites) <= 10: return

    satellites['SubhaloMass'] = pd.to_numeric(satellites['SubhaloMass'], errors='coerce').fillna(0)
    top_10_sats = satellites.sort_values(by='SubhaloMass', ascending=False).head(10)
    
    # Recombine
    df_top_10_pos = pd.concat([central_row, top_10_sats], ignore_index=True)
    
    # Get the RELATIVE INDICES required to fetch from the tracks file
    target_track_indices = top_10_sats['relative_track_id'].tolist()
    
    # Drop the helper column before saving to keep format clean
    df_to_save = df_top_10_pos.drop(columns=['relative_track_id'])
    df_to_save.to_csv(os.path.join(base_dir, f"{base_name}_positions_top_10.csv"), index=False)
    
    rel_pos = df_top_10_pos.iloc[1:][['x', 'y', 'z']].values.astype(np.float64)
    
    r200 = 0.0
    if 'Group_R_Crit200' in central_row.columns:
        r200 = float(central_row['Group_R_Crit200'].iloc[0])
    
    try:
        met_centroid = metrics.Centroid.eval(rel_pos)
        met_mrl = metrics.MRL.eval(rel_pos)
        met_pairwise = metrics.Pairwise.eval(rel_pos) if len(rel_pos) > 1 else np.nan
        met_hemi = metrics.Hemi.eval(rel_pos, num_grid=30)[0] 
        met_rms = calc_rms_normalized(rel_pos, r200)
    except Exception as e:
        print(f"Error calculating metrics for {base_name}: {e}")
        return

    metrics_row = {
        'centroid': met_centroid,
        'mrl': met_mrl,
        'pairwise': met_pairwise,
        'hemi_contrast': met_hemi,
        'normalised_rms_distance': met_rms,
        'subhalo_count': len(rel_pos),
        'HaloID': base_name.split('halo_')[1].split('_')[0],
        'SnapNum': base_name.split('snap_')[1].split('_')[0]
    }
    
    pd.DataFrame([metrics_row]).to_csv(os.path.join(base_dir, f"{base_name}_metrics_top_10.csv"), index=False)

    tracks_file = os.path.join(base_dir, f"{base_name}_all_tracks.csv")
    if not os.path.exists(tracks_file): return

    try:
        df_tracks = pd.read_csv(tracks_file)
    except: return

    # FIX: Filter using the 'relative_track_id' we mapped earlier
    df_tracks_top_10 = df_tracks[df_tracks['subhalo_index'].isin(target_track_indices)].copy()
    
    df_tracks_top_10.to_csv(os.path.join(base_dir, f"{base_name}_all_tracks_top_10.csv"), index=False)
    
    if df_tracks_top_10.empty:
        return

    # Group by subhalo to define time steps
    grouped = df_tracks_top_10.groupby('subhalo_index')
    # Create dict: {track_id : numpy array of positions}
    track_dict = {name: group[['x', 'y', 'z']].values for name, group in grouped}
    
    # Determine the deepest history length
    max_len = max([len(v) for v in track_dict.values()]) if track_dict else 0
    
    history_metrics = {
        'SnapNum': [], # Simulated SnapNum
        'Centroid_Distance': [], 'MRL': [], 'Pairwise': [], 
        'Hemi_Contrast': [], 'Normalised_RMS_Distance': []
    }
    
    current_snap = int(metrics_row['SnapNum'])

    # Iterate backwards through history steps
    # We assume track arrays are ordered [Oldest -> Newest] or [Newest -> Oldest]
    # Standard reader.py usually puts latest position at end of array.
    # We process index `i` from the END of the arrays to align "Present Day".
    
    for i in range(max_len):
        step_positions = []
        
        # For each of our top 10 satellites
        for idx in target_track_indices:
            if idx in track_dict:
                track_arr = track_dict[idx]
                
                # Check if this satellite exists at this time step 'i' backwards from now
                # Index from end: len - 1 - i
                arr_idx = len(track_arr) - 1 - i
                
                if arr_idx >= 0:
                    step_positions.append(track_arr[arr_idx])
        
        # Need at least 2 satellites to calculate meaningful metrics
        if len(step_positions) < 2: continue
            
        step_pos_arr = np.array(step_positions, dtype=np.float64)
        
        try:
            # Approximate SnapNum: Current - i
            sim_snap = current_snap - i
            if sim_snap < 0: break # Sanity check
            
            # Note: Centroid.eval returns scalar or vector depending on implementation
            # Since metrics.py is imported, we handle return types carefully
            c_dist = metrics.Centroid.eval(step_pos_arr)
            if isinstance(c_dist, np.ndarray): c_dist = np.linalg.norm(c_dist)
            
            hemi_val = metrics.Hemi.eval(step_pos_arr, num_grid=30)[0]

            history_metrics['SnapNum'].append(sim_snap)
            history_metrics['Centroid_Distance'].append(c_dist)
            history_metrics['MRL'].append(metrics.MRL.eval(step_pos_arr))
            history_metrics['Pairwise'].append(metrics.Pairwise.eval(step_pos_arr))
            history_metrics['Hemi_Contrast'].append(hemi_val)
            history_metrics['Normalised_RMS_Distance'].append(calc_rms_normalized(step_pos_arr, r200))
        except:
            continue

    if len(history_metrics['SnapNum']) > 0:
        # Sort by SnapNum ascending for plotting (Oldest -> Newest)
        df_evol = pd.DataFrame(history_metrics).sort_values('SnapNum')
        df_evol.to_csv(os.path.join(base_dir, f"{base_name}_metric_evolution_top_10.csv"), index=False)

def main():
    print("--- Top 10 Subhalo Processor (Recursive & Fixed) ---")
    
    folder_path = INPUT_FOLDER_PATH.strip('"')
    
    if not os.path.isdir(folder_path):
        print(f"Invalid directory: {folder_path}")
        return

    # Filter out files we created or dark files
    search_pattern = os.path.join(folder_path, "**", "*_positions.csv")
    pos_files = glob.glob(search_pattern, recursive=True)
    pos_files = [f for f in pos_files if "_top_10" not in f and "dark" not in f]
    
    print(f"Found {len(pos_files)} candidate files.")
    print("Starting processing...")
    
    count = 0
    for f in tqdm(pos_files):
        process_single_snap(f)
        count += 1
        
    print(f"Done! Processed {count} files.")

if __name__ == "__main__":
    main()