import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import illustris_python as il
import glob


def read_central_subhalo_tracks_as_columns(base_folder_path, min_snap=68):
    """
    Reads and pivots central ID files from all Halo subfolders directly within
    a specific snapshot folder (e.g., Snap99). The category distinction is removed.

    Args:
        base_folder_path (str): Path to the specific Snap folder (e.g., .../Snap99).
        min_snap (int): Minimum snapshot number to include in the tracks.
    """

    # Look for files under Halo ID folders directly under SnapXX
    search_pattern = os.path.join(base_folder_path, "Halo*", "*_central_ids.csv")
    all_central_id_files = glob.glob(search_pattern, recursive=False)

    if not all_central_id_files:
        print(f"No central_ids.csv files found directly under: {base_folder_path}/Halo*")
        return pd.DataFrame(), pd.Series()

    all_halo_dfs = []
    category_map = {} # To store the category for each HaloID_at_Snap99

    desc = f"Reading SubfindID tracks from {os.path.basename(base_folder_path)} (all Tracked)"
    for file_path in tqdm(all_central_id_files, desc=desc):
        try:
            filename = os.path.basename(file_path)
            # Example filename: sim_TNG100-1_halo_488_snap_99_central_ids.csv
            halo_id_str = filename.split('_halo_')[1].split('_snap_')[0]
            current_halo_id = int(halo_id_str)

            df_track = pd.read_csv(file_path)
            df_filtered = df_track[df_track['SnapNum'] >= min_snap].copy()
            df_filtered['Current_Halo_ID'] = current_halo_id
            df_filtered['SnapNum_str'] = 'Snap_' + df_filtered['SnapNum'].astype(str)

            # Ensure the index is the current halo ID, columns are snapshots
            df_pivot = df_filtered.pivot(index='Current_Halo_ID', columns='SnapNum_str', values='SubfindID')

            all_halo_dfs.append(df_pivot)
            category_map[current_halo_id] = 'Tracked'

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    if not all_halo_dfs:
        return pd.DataFrame(), pd.Series()

    final_df = pd.concat(all_halo_dfs).sort_index()

    # Sort columns by snapshot number
    snap_cols = [col for col in final_df.columns if col.startswith('Snap_')]
    sorted_snap_cols = sorted(snap_cols, key=lambda x: int(x.split('_')[1]))

    # Create the category Series
    category_series = pd.Series(category_map).sort_index()

    return final_df[sorted_snap_cols], category_series


def create_halo_tracking_df(central_subhalo_tracks_df, base_sim_path, snaps_to_skip=set()):
    """
    Maps central SubfindIDs to their parent HaloIDs for snapshots that have not been processed yet.
    Defaults to -1 for missing data.
    """
    # Initialize a new DataFrame for HaloIDs with -1 as the default value
    halo_tracks_df = pd.DataFrame(index=central_subhalo_tracks_df.index,
                                  columns=central_subhalo_tracks_df.columns,
                                  dtype=np.int64).fillna(-1)

    snap_numbers = [int(col.split('_')[1]) for col in central_subhalo_tracks_df.columns]
    for snap in tqdm(snap_numbers, desc="Mapping Centrals to Halos"):
        # --- Skip snapshots that are already processed ---
        if snap in snaps_to_skip:
            print(f"\n[INFO] Snap {snap} found in existing summary. Skipping processing.")
            continue

        try:
            halos_data = il.groupcat.loadHalos(base_sim_path, snap, fields=['GroupFirstSub'])
            group_first_sub_array = halos_data['GroupFirstSub'] if isinstance(halos_data, dict) else halos_data
            if group_first_sub_array is None or group_first_sub_array.size == 0: continue

            temp_df = pd.DataFrame({'HaloID': np.arange(len(group_first_sub_array)), 'GroupFirstSub': group_first_sub_array})
            unique_df = temp_df.drop_duplicates(subset='GroupFirstSub', keep='first')
            inverse_map = pd.Series(unique_df.HaloID.values, index=unique_df.GroupFirstSub)

            # Map SubfindID to HaloID, filling any misses with -1
            mapped_ids = central_subhalo_tracks_df[f'Snap_{snap}'].map(inverse_map).fillna(-1).astype(np.int64)
            halo_tracks_df[f'Snap_{snap}'] = mapped_ids

        except FileNotFoundError:
            print(f"\n[INFO] Raw simulation data for Snap {snap} not found. Halo IDs will be -1.")
        except Exception as e:
            print(f"\n[ERROR] Could not process Snap {snap}: {e}. Halo IDs will be -1.")

    return halo_tracks_df


def check_export_status(snap_no, halo_id, base_export_path):
    """
    Checks if a halo's export folder exists, looking *directly* under the SnapXX folder.
    This is consistent with the new flat directory structure.
    """
    if pd.isna(halo_id) or halo_id == -1: return False

    # --- MODIFIED CHECK: Search directly under SnapXX for the Halo ID folder ---
    snap_folder = os.path.join(base_export_path, f"Snap{snap_no}")
    search_path = os.path.join(snap_folder, f"Halo {int(halo_id)}")

    # Use os.path.isdir as the expected structure is now: SnapXX/Halo ID/
    return os.path.isdir(search_path)


base_export_folder = "halo_exports/TNG100-1"
base_sim_path = r"D:/Abhilash_Thesis_Data/Simulation_Data/TNG100-1/output"
output_filename = "halo_lineage_summary.csv"
output_path = os.path.join(base_export_folder, output_filename)

existing_summary_df = None
processed_snaps = set()
if os.path.exists(output_path):
    print(f"--- Found existing summary file at: {output_path} ---")

    # Keep the index as 'HaloID_at_Snap99' if it exists, otherwise rely on default index
    try:
        existing_summary_df = pd.read_csv(output_path, index_col="HaloID_at_Snap99")
    except ValueError:
         print("Warning: Existing summary file is missing 'HaloID_at_Snap99' index column. Reading without index.")
         existing_summary_df = pd.read_csv(output_path)


    for col in existing_summary_df.columns:
        if col.startswith("HaloID_Snap_"):
            # A snap is "processed" if its HaloID column has at least one valid ID (i.e., not -1)
            if (existing_summary_df[col] > -1).any():
                # Extract the number from 'HaloID_Snap_99'
                try:
                    snap_num = int(col.split('_')[-1])
                    processed_snaps.add(snap_num)
                except:
                    continue # Skip columns that don't match the expected naming convention

    if processed_snaps:
        print(f"Snapshots with valid Halo IDs found: {sorted(list(processed_snaps))}. These will be skipped.\n")
    else:
        print("Existing summary contains no processed snapshots. Will process all available snapshots.\n")


MIN_SNAP_TO_CONSIDER = 68
all_snap_folders = glob.glob(os.path.join(base_export_folder, "Snap*"))
available_snap_nums = sorted([int(os.path.basename(f).split('Snap')[-1]) for f in all_snap_folders])
print(f"Discovered all available snapshot folders: {available_snap_nums}")


snap99_export_folder = os.path.join(base_export_folder, "Snap99")
subfind_id_df, category_series = read_central_subhalo_tracks_as_columns(
    snap99_export_folder,
    min_snap=MIN_SNAP_TO_CONSIDER
)

# --- Ensure the DataFrame has columns for all available snapshots ---
all_snap_cols_expected = [f"Snap_{s}" for s in available_snap_nums if s >= MIN_SNAP_TO_CONSIDER]
subfind_id_df = subfind_id_df.reindex(columns=all_snap_cols_expected, fill_value=np.nan)


if not subfind_id_df.empty:
    new_halo_id_df = create_halo_tracking_df(subfind_id_df, base_sim_path, snaps_to_skip=processed_snaps)

    final_summary_df = pd.DataFrame(index=subfind_id_df.index)
    final_summary_df.index.name = "HaloID_at_Snap99"

    # The value is now consistently 'Tracked' as the category folder distinction is removed
    final_summary_df['Halo_Category_Snap99'] = category_series

    # Use the complete list of expected columns to build the final CSV
    for col_name in all_snap_cols_expected:
        snap_num = int(col_name.split('_')[1])

        final_summary_df[f"SubfindID_{col_name}"] = subfind_id_df[col_name]

        halo_id_col = f"HaloID_{col_name}"
        if (existing_summary_df is not None and
            halo_id_col in existing_summary_df.columns and
            snap_num in processed_snaps):

            # Use existing data, ensuring alignment on the index
            if existing_summary_df.index.name == "HaloID_at_Snap99":
                final_summary_df[halo_id_col] = existing_summary_df[halo_id_col]
            else:
                 # If index is missing, we must re-calculate the HaloID to be safe, so we skip reading old data here
                 final_summary_df[halo_id_col] = new_halo_id_df[col_name]
        else:
            final_summary_df[halo_id_col] = new_halo_id_df[col_name]

        # The check_export_status function has been updated to search the new flat structure
        exported_col = f"Exported_{col_name}"
        final_summary_df[exported_col] = final_summary_df[halo_id_col].apply(
            lambda halo_id: check_export_status(snap_num, halo_id, base_export_folder)
        )

    # Reorder columns for readability: Category first, then Snap_99 (HaloID, SubfindID, Exported)
    ordered_columns = ['Halo_Category_Snap99']

    # Get all column names related to the latest snapshot (Snap99) first
    snap99_cols = [col for col in final_summary_df.columns if col.endswith('_Snap_99')]
    # Sort them: HaloID, SubfindID, Exported
    snap99_cols_ordered = sorted(snap99_cols, key=lambda x: x.split('_')[0], reverse=True)
    ordered_columns.extend(snap99_cols_ordered)

    # Get all other columns and append
    other_cols = [col for col in final_summary_df.columns if col not in ordered_columns]

    # Sort the remaining columns by SnapNum, then by metric type (HaloID, SubfindID, Exported)
    other_cols_sorted = sorted(other_cols,
                               key=lambda x: (int(x.split('_')[-1]), x.split('_')[0]))

    ordered_columns.extend(other_cols_sorted)

    # Reindex the DataFrame with the new column order
    final_summary_df = final_summary_df.reindex(columns=ordered_columns)


    final_summary_df.to_csv(output_path)

    print("\n--- Summary ---")
    print(f"✅ Updated lineage summary for {len(final_summary_df)} halos, with category 'Tracked', has been saved to:")
    print(output_path)

else:
    print("❌ No subhalo tracks found. Summary file was not created/updated.")
