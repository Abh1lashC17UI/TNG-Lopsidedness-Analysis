import os
import glob
import re
import pandas as pd
from tqdm import tqdm

INPUT_FOLDER_PATH = r"C:\Users\Abhilash\Documents\Thesis\halo_exports\TNG50-1"
OUTPUT_FILENAME = "top_10_systems_summary.csv"

def generate_summary():
    print("--- Top 10 Systems Summary Generator ---")
    
    folder_path = INPUT_FOLDER_PATH.strip('"')
    
    if not os.path.isdir(folder_path):
        print(f"Invalid directory: {folder_path}")
        return

    print(f"Scanning for Top 10 files in:\n{folder_path}")
    
    # Find all files indicating a Top 10 system exists
    search_pattern = os.path.join(folder_path, "**", "*_positions_top_10.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print("No Top 10 files found. Run process_top_10.py first.")
        return

    # Dictionary to store data: { SnapNum : [HaloID, HaloID, ...] }
    snap_data = {}
    
    # Regex: Capture Halo ID and Snap Num
    filename_pattern = re.compile(r"halo_(\d+)_snap_(\d+)_")

    for filepath in tqdm(files, desc="Parsing"):
        filename = os.path.basename(filepath)
        match = filename_pattern.search(filename)
        
        if match:
            halo_id = int(match.group(1))
            snap_num = int(match.group(2))
            
            if snap_num not in snap_data:
                snap_data[snap_num] = []
            
            snap_data[snap_num].append(halo_id)

    # Convert to DataFrame
    rows = []
    for snap in sorted(snap_data.keys(), reverse=True):
        halos = sorted(snap_data[snap])
        rows.append({
            'SnapNum': snap,
            'System_Count': len(halos),
            'Halo_IDs': str(halos) # Store list as string representation
        })
    
    df = pd.DataFrame(rows)
    
    # Save
    output_path = os.path.join(folder_path, OUTPUT_FILENAME)
    df.to_csv(output_path, index=False)
    
    print(f"\nSummary saved to:\n{output_path}")
    print("-" * 30)
    print(df.head())
    print("-" * 30)

if __name__ == "__main__":
    generate_summary()