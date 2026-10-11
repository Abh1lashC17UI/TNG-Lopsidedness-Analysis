import os
import shutil
import re

'''
Sorts the exported csv files into folders for organisation and easier reading
'''

def organize_halo_folders(source_dir):
    """
    Sorts the exported csv files into folders for organisation and easier reading.
    Takes the target directory path as an argument.
    """
    print(f"\n--- Organizing files in: {source_dir} ---")
    
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} does not exist. Skipping organization.")
        return

    # List only files in the source directory (ignores already created folders)
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Regex to capture halo number (e.g., 'sim_TNG100-1_halo_34_snap_99.csv' -> '34')
    pattern = re.compile(r'halo_(\d+)_')
    
    moved_count = 0
    for filename in files:
        match = pattern.search(filename)
        if match:
            halo_num = match.group(1)
            
            # Create the new folder name
            folder_name = f"Halo {halo_num}"
            folder_path = os.path.join(source_dir, folder_name)
            
            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Move the file into the folder
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(folder_path, filename)
            shutil.move(src_path, dst_path)
            moved_count += 1
            
    print(f"Successfully organized {moved_count} files into Halo subfolders.\n")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        organize_halo_folders(sys.argv[1])

    else:
        # Folder which I want to change
        source_dir = r"halo_exports\Snap99"

        organize_halo_folders(source_dir)
