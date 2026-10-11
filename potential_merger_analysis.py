import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

sim_name = "TNG_100"

BASE_DIR = r"E:\Masters\Master's Thesis\Code\Illustris-TNG\halo_exports\Save_"+sim_name+r"-1_by_volume_PBC"
LINEAGE_FILE = os.path.join(BASE_DIR, "halo_lineage_summary.csv")

classification_filename  = "Top_Bottom_Systems_Snap99_Metrics_20251224.csv" if sim_name=="TNG_100" else "Top_Bottom_Systems_Snap99_Metrics_20260101.csv"
CLASSIFICATION_FILE = os.path.join(BASE_DIR, classification_filename)

# Snapshot Range
START_SNAP = 68
END_SNAP = 99

WINDOW_SIZE = 3
MIN_ABS_CHANGE = 5
MIN_REL_CHANGE = 1.4

def load_ids_from_csv(csv_path, metric_name='MRL'):
    if not os.path.exists(csv_path):
        print(f"Error: Classification file not found at {csv_path}")
        return [], []

    df = pd.read_csv(csv_path)
    row = df[df['Metric'] == metric_name]
    if row.empty: return [], []
    row = row.iloc[0]

    def parse(s):
        if pd.isna(s) or str(s).strip() == '': return []
        return [int(x) for x in str(s).split(', ') if x]

    return parse(row['Most Lopsided (Top 10%) Halo IDs (Snap 99)']), \
           parse(row['Least Lopsided (Bottom 10%) Halo IDs (Snap 99)'])

def get_satellite_count_local(snap, halo_id):
    halo_dir = os.path.join(BASE_DIR, f"Snap{snap}", f"Halo {halo_id}")
    if not os.path.exists(halo_dir): return np.nan

    # Find visible positions file
    files = glob.glob(os.path.join(halo_dir, "*_positions.csv"))
    visible_files = [f for f in files if "dark" not in os.path.basename(f).lower()]

    if not visible_files: return np.nan

    try:
        # Subtract 1 because file includes Central
        count = len(pd.read_csv(visible_files[0])) - 1
        return max(0, count)
    except:
        return np.nan

def detect_step_change(halo_id, snaps, counts, debug=False):
    """
    Scans the timeline for sustained jumps.
    Includes a COOLDOWN to prevent multiple flags for the same event.
    """
    WINDOW = 5

    # Once an event is found, we skip this many snapshots to let the window pass.
    # Usually equal to WINDOW size is a safe bet.
    COOLDOWN_STEPS = 5

    if len(counts) < 2 * WINDOW:
        return []

    events = []

    # Using a while loop or a skip-tracker to handle the cooldown
    skip_until_index = -1

    for i in range(WINDOW, len(counts) - WINDOW):

        if i < skip_until_index:
            continue

        pre_window = counts[i-WINDOW : i]
        post_window = counts[i : i+WINDOW]

        val_pre = np.median(pre_window)
        val_post = np.median(post_window)

        diff = val_post - val_pre
        # Add epsilon to prevent division by zero
        ratio = val_post / (val_pre + 0.1)

        event_found = False
        event_type = None

        if diff >= MIN_ABS_CHANGE and ratio >= MIN_REL_CHANGE:
            event_type = "Infall"
            event_found = True

        elif diff <= -MIN_ABS_CHANGE and (val_pre / (val_post + 0.1)) >= MIN_REL_CHANGE:
            event_type = "Loss"
            event_found = True

        if event_found:
            if debug:
                print(f"  [!] Halo {halo_id} {event_type} at Snap {snaps[i]}:")
                print(f"      Pre: {val_pre} -> Post: {val_post} | Diff: {diff}")

            events.append((snaps[i], event_type, diff))

            # Apply Cooldown: Skip analysis for the next few snapshots
            skip_until_index = i + COOLDOWN_STEPS

    return events

def main():
    print("--- Local Sustained Change Detector ---")

    lineage_df = pd.read_csv(LINEAGE_FILE, index_col="HaloID_at_Snap99")
    top_ids, bot_ids = load_ids_from_csv(CLASSIFICATION_FILE)

    all_targets = top_ids + bot_ids
    results = {} # Store events: {halo_id: [events]}
    histories = {} # Store raw data for plotting

    print(f"Scanning {len(all_targets)} systems for step changes...")

    for snap99_id in tqdm(all_targets):
        if snap99_id not in lineage_df.index: continue

        # Build History
        lineage_row = lineage_df.loc[snap99_id]
        snaps, counts = [], []

        for snap in range(START_SNAP, END_SNAP + 1):
            col = f"HaloID_Snap_{snap}"
            if col in lineage_row and lineage_row[col] != -1:
                c = get_satellite_count_local(snap, int(lineage_row[col]))
                if not np.isnan(c):
                    snaps.append(snap)
                    counts.append(c)

        # Detect Steps (Enable Debug for the first few to check)
        if snaps:
            events = detect_step_change(snap99_id, snaps, counts, debug=False)

            if events:
                results[snap99_id] = events
            histories[snap99_id] = (snaps, counts)

    # --- STATISTICS ---
    top_infall, top_loss = 0, 0
    bot_infall, bot_loss = 0, 0

    # Analyze Top (Lopsided)
    for uid in top_ids:
        if uid in results:
            evs = results[uid]
            if any(e[1] == "Infall" for e in evs): top_infall += 1
            if any(e[1] == "Loss" for e in evs): top_loss += 1

    # Analyze Bottom (Isotropic)
    for uid in bot_ids:
        if uid in results:
            evs = results[uid]
            if any(e[1] == "Infall" for e in evs): bot_infall += 1
            if any(e[1] == "Loss" for e in evs): bot_loss += 1

    top_frac = top_infall / len(top_ids) if len(top_ids) > 0 else 0.0
    bot_frac = bot_infall / len(bot_ids) if len(bot_ids) > 0 else 0.0

    print("\n" + "="*40)
    print("SUSTAINED CHANGE STATISTICS")
    print("="*40)
    print(f"{'Group':<15} | {'Infall Events':<15} | {'Loss Events':<15} | {'Total Systems'}")
    print("-" * 60)

    print(f"{'Lopsided':<15} | {top_infall:<15} | {top_loss:<15} | {len(top_ids)}")
    print(f"{'Isotropic':<15} | {bot_infall:<15} | {bot_loss:<15} | {len(bot_ids)}")

    print(f"\nInfall Fraction: Top {top_frac:.2f} vs Bottom {bot_frac:.2f}")

    # --- PLOTTING INTERESTING CASES ---

    # Using plt.get_cmap to avoid warning
    colors = plt.get_cmap('tab20', 20)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    color_idx = 0
    count = 0

    for uid in top_ids:
        if uid in histories:
            s, c = histories[uid]

            # Check for Infall
            has_infall = uid in results and any(e[1] == "Infall" for e in results[uid])

            if has_infall:
                # Interesting: Unique Color + Label
                line_color = colors(color_idx % 20)
                color_idx += 1
                lbl = f"Halo {uid}"
                alpha = 0.8
                lw = 2.0 # Thicker to stand out
                z_ord = 5
            else:
                # Boring: Faint Red Background
                line_color = 'red'
                lbl = None
                alpha = 0.05
                lw = 1.0
                z_ord = 1

            # Plot the line
            axes[0].plot(s, c, alpha=alpha, color=line_color, linewidth=lw, label=lbl, zorder=z_ord)

            # Mark the event if interesting
            if has_infall:
                for e in results[uid]:
                    if e[1] == "Infall":
                        y_val = np.interp(e[0], s, c)
                        axes[0].scatter(e[0], y_val, color='black', marker='^', zorder=10, s=40)

            count += 1
            if count > 300: break

    axes[0].set_title(f"Lopsided Systems: Sustained Infall (Detected in {top_infall} systems)")
    axes[0].set_ylabel("Satellite Count")
    axes[0].grid(True, alpha=0.3)

    # Legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small', title="Interesting Lopsided")

    color_idx = 0
    count = 0

    for uid in bot_ids:
        if uid in histories:
            s, c = histories[uid]

            has_infall = uid in results and any(e[1] == "Infall" for e in results[uid])

            if has_infall:
                # Interesting: Unique Color
                line_color = colors(color_idx % 20)
                color_idx += 1
                lbl = f"Halo {uid}"
                alpha = 0.8
                lw = 2.0
                z_ord = 5
            else:
                # Boring: Faint Blue Background
                line_color = 'blue'
                lbl = None
                alpha = 0.05
                lw = 1.0
                z_ord = 1

            axes[1].plot(s, c, alpha=alpha, color=line_color, linewidth=lw, label=lbl, zorder=z_ord)

            if has_infall:
                for e in results[uid]:
                     if e[1] == "Infall":
                        y_val = np.interp(e[0], s, c)
                        axes[1].scatter(e[0], y_val, color='black', marker='^', zorder=10, s=40)

            count += 1
            if count > 300: break

    axes[1].set_title(f"Isotropic Systems: Mostly Stable (Infall in {bot_infall} systems)")
    axes[1].set_xlabel("Snapshot")
    axes[1].set_ylabel("Satellite Count")
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[1].get_legend_handles_labels()
    if labels:
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small', title="Interesting Isotropic")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
