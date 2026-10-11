import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import illustris_python as il
from classes import Halo
import glob

def export_halos(snap_num, halos_to_export_map, base_path, sim_name, base_export_folder):
    """
    Loads data for a single snapshot and processes a map of {halo_id: category}.

    Args:
        snap_num (int): The snapshot number to export from.
        halos_to_export_map (dict): A dictionary mapping {HaloID_at_SnapXX: Halo_Category_Snap99}.
                                    The category is now unused for folder creation.
        base_path (str): Path to the raw simulation output.
        sim_name (str): Name of the simulation (e.g., 'TNG100-1').
        base_export_folder (str): The root path where data is saved.
    """
    halo_ids = list(halos_to_export_map.keys())
    print(f"\n--- Preparing to Export {len(halo_ids)} Halos from Snapshot {snap_num} ---")

    # Load Simulation Data for the given snapshot
    print(f"Loading simulation data for Snapshot {snap_num}...")
    try:
        subhalo_fields = ['SubhaloFlag', 'SubhaloBHMdot', 'SubhaloPos', 'SubhaloParent', 'SubhaloGrNr', 'SubhaloMass', 'SubhaloStellarPhotometrics', 'SubhaloStarMetalFractions', 'SubhaloVel']
        subhalos = il.groupcat.loadSubhalos(base_path, snap_num, subhalo_fields)
        halo_fields = ['GroupCM', 'GroupPos', 'GroupFirstSub', 'GroupNsubs', 'GroupMass', 'Group_R_Crit200']
        halos = il.groupcat.loadHalos(base_path, snap_num, halo_fields)
        header = il.groupcat.loadHeader(base_path, snapNum=snap_num)
        lbox = header['BoxSize']
        subhalo_pos = subhalos['SubhaloPos']
        print("Data loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load simulation data for Snapshot {snap_num}: {e}")
        return

    # Process and Export Each Specified Halo
    for halo_id in tqdm(halo_ids, desc=f"Exporting from Snap {snap_num}"):

        # Retrieve the category (now 'Tracked' or similar), though it is no longer used for the path.

        # Path is now: base_export_folder/SnapXX/
        # The Halo.save() method will automatically create the final 'Halo ID' folder.
        final_export_path = os.path.join(base_export_folder, f"Snap{snap_num}")

        try:
            halo_to_export = Halo(halos, halo_id, halo_fields, lbox, snap_num, sim_name)

            # Safety check: ensure the halo has a central subhalo index
            if halo_id >= len(halos['GroupFirstSub']):
                print(f"\n[ERROR] Halo ID {halo_id} is out of bounds for the current snapshot {snap_num}. Skipping.")
                continue

            central_subhalo_index = halos['GroupFirstSub'][halo_id]
            halo_to_export.get_subhalos_by_volume(subhalos, subhalo_fields, central_subhalo_index, subhalo_pos, force_rerun=True)

            # Perform calculations
            halo_to_export.calc_metrics()

            # Halo.save() takes 'outdir', which is the path *above* the final 'Halo ID' folder.
            halo_to_export.save(
                outdir=final_export_path, # Passed as full path: base_export_folder/SnapXX/
                save_metrics=True,
                save_subhalo_pos=True,
                save_dark_subhalo=True,
                save_tracks=False,
                save_metric_evolution_using_tracks=False,
                save_central_ids=True
                )

        except Exception as e:
            # The error message is updated to reflect the new, simpler path
            print(f"\n[ERROR] Failed to process Halo {halo_id} from Snap {snap_num} into {os.path.normpath(final_export_path)}/Halo {halo_id}: {e}")
            continue

    print(f"\n--- Finished Export for Snapshot {snap_num} ---")


def main():
    base_export_folder = "halo_exports/TNG50-1"
    base_sim_path = r"D:/Abhilash_Thesis_Data/Simulation_Data/TNGG50-1/output"
    sim_name = 'TNG50-1'
    lineage_file_path = os.path.join(base_export_folder, "halo_lineage_summary.csv")

    if not os.path.exists(lineage_file_path):
        print(f"[ERROR] Lineage summary file not found at: {lineage_file_path}")
        print("Please run the 'intersting_halo_finder.py' script first to generate it.")
        return

    print(f"Reading lineage data from {lineage_file_path}...")
    try:
        lineage_df = pd.read_csv(lineage_file_path, index_col="HaloID_at_Snap99")
    except ValueError:
        print("[ERROR] Lineage summary file is malformed or missing the required index 'HaloID_at_Snap99'.")
        return

    # Check for the mandatory category column
    if 'Halo_Category_Snap99' not in lineage_df.columns:
        print("[ERROR] The 'halo_lineage_summary.csv' is missing the 'Halo_Category_Snap99' column.")
        print("Please run the latest version of 'intersting_halo_finder.py' first.")
        return


    halos_to_export_by_snap = {} # Stores {snap_num: {HaloID_at_SnapXX: Halo_Category_Snap99}}

    for col in lineage_df.columns:
        if col.startswith("Exported_Snap_"):
            snap_num = int(col.split('_')[-1])
            halo_id_col = f"HaloID_Snap_{snap_num}"

            # Filter for halos that are NOT exported, and have a valid HaloID at that snapshot
            unexported_series = lineage_df[(lineage_df[col] == False) & (lineage_df[halo_id_col] > -1)][halo_id_col]

            if not unexported_series.empty:
                # Create a temporary DataFrame to hold the two crucial columns:
                temp_df = pd.DataFrame({
                    'HaloID_at_CurrentSnap': unexported_series.astype(int),
                    'Halo_Category_Snap99': lineage_df.loc[unexported_series.index, 'Halo_Category_Snap99']
                })

                # The final map needed for export_halos is {HaloID_at_CurrentSnap: Halo_Category_Snap99}
                export_map = temp_df.set_index('HaloID_at_CurrentSnap')['Halo_Category_Snap99'].to_dict()

                halos_to_export_by_snap[snap_num] = export_map

    while True:
        if not halos_to_export_by_snap:
            print("\n✅ All halos in the lineage file are currently exported.")
        else:
            print("\n----------------------------------------------------")
            print("⚠️ Found the following halos that need to be exported:")
            for snap, export_map in sorted(halos_to_export_by_snap.items()):
                print(f"  - Snapshot {snap}: {len(export_map)} halos")
            print("----------------------------------------------------")

        print("\nWhat would you like to do?")
        print("  1. Export ALL pending halos listed above")
        print("  2. Export halos for a SPECIFIC snapshot from the list")
        # --- Option 3 updated to reflect the removal of the category concept ---
        print("  3. Manually export specific Halo IDs (will be marked as 'Tracked')")
        print("  4. Exit")
        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice == '1':
            if not halos_to_export_by_snap:
                print("There are no pending halos to export.")
                continue
            for snap_num, export_map in sorted(halos_to_export_by_snap.items()):
                export_halos(snap_num, export_map, base_sim_path, sim_name, base_export_folder)
            print("\nFinished exporting all pending halos. The list below may now be empty.")

        elif choice == '2':
            if not halos_to_export_by_snap:
                print("There are no pending halos to export.")
                continue
            try:
                snap_keys = list(sorted(halos_to_export_by_snap.keys()))
                snap_choice = int(input(f"Enter the snapshot number to export from {snap_keys}: ").strip())
                if snap_choice in halos_to_export_by_snap:
                    export_map = halos_to_export_by_snap[snap_choice]
                    export_halos(snap_choice, export_map, base_sim_path, sim_name, base_export_folder)
                    print(f"\nFinished exporting for Snapshot {snap_choice}.")
                else:
                    print("[ERROR] Invalid snapshot number. Please choose from the list.")
            except ValueError:
                print("[ERROR] Please enter a valid number.")

        # --- Handles a comma-separated list of Halo IDs (manual export to 'Tracked' category) ---
        elif choice == '3':
            try:
                snap_manual = int(input("Enter the Snapshot Number to export from: ").strip())

                # Get comma-separated string of IDs
                halos_str = input(f"Enter comma-separated Halo IDs to export from Snapshot {snap_manual} (e.g., 488, 599, 627): ").strip()

                # Process the string into a list of integers
                if halos_str:
                    halo_ids_manual = [int(s.strip()) for s in halos_str.split(',')]

                    # For manual entry, we use the default category 'Tracked'
                    manual_export_map = {hid: 'Tracked' for hid in halo_ids_manual}

                    print(f"\nPreparing to manually export {len(halo_ids_manual)} halos (to the root Snap folder): {halo_ids_manual} from Snapshot {snap_manual}...")
                    export_halos(snap_manual, manual_export_map, base_sim_path, sim_name, base_export_folder)
                    print(f"\nFinished manually exporting specified halos.")
                else:
                    print("[INFO] No Halo IDs entered. Returning to menu.")

            except ValueError:
                print("[ERROR] Invalid input. Please enter a number for the snapshot and a comma-separated list of numbers for Halo IDs.")

        elif choice == '4':
            print("Exiting.")
            break

        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, 3, or 4.")

        # Re-read the summary file to update the pending list after an export run
        if 'choice' in locals() and choice in ['1', '2', '3']:
            print("\nChecking export status...")
            main()
            return

        print("\nReturning to the main menu...")


main()
