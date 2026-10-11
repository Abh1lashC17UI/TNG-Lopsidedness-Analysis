import re
import sys
import os
import glob
import time
from contextlib import contextmanager
import pandas as pd
import numpy as np
from tqdm import tqdm
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QLabel, QListWidget, QTextEdit, QCheckBox,
                             QMessageBox, QTabWidget, QComboBox, QGroupBox, QRadioButton, QSlider, QGridLayout,
                             QScrollArea, QSplitter, QFrame, QSizePolicy, QAbstractItemView, QTableWidget,
                             QTableWidgetItem, QHeaderView, QTreeWidget, QTreeWidgetItem)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QStandardItem, QPixmap, QImage, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from analysis import (
    aggregate_host_metric_rows,
    aggregate_satellite_metric_rows,
    build_contingency_inputs_from_top_bottom,
    build_system_table_dataframe,
    compare_distributions,
    compute_contingency_analysis,
    compute_correlation_matrices,
    compute_merger_features,
    compute_percentile_band,
    compute_percentile_band_from_matrix,
    parse_top10_reference,
    process_raw_overview_data,
)
from p_value_engine import (
    IsotropicNullEngine,
    compute_p_iso,
    compute_p_cosmo,
    build_cosmo_ensemble,
    invalidate_cosmo_ensemble_cache,
    _compute_d_cen_null,
    N_ITER_DEFAULT,
)

# Canvas for visualising the system
class Plot3DCanvas(FigureCanvas):
    def __init__(self, app_instance, parent=None):
        fig = Figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.cbar = None
        self.sm = None
        self.focused_track_id = None  # None = default view
        self.track_data = {}  # Store all plotted track data
        self.central_pos = np.array([0, 0, 0])
        self.central_mass = 50.
        self.pick_event_cid = None
        self.dark_info_text = None
        self.dark_subhalo_points = []
        self.group_r_crit200 = 0.0

        if not hasattr(self.ax, 'get_proj_type'):
            self.ax.get_proj_type = lambda: 'ortho' if getattr(self.ax, '_focal_length', 1) == np.inf else 'persp'

        self.app_instance = app_instance
        super().__init__(fig)
        self._apply_dark_mpl_style_3d()
        if parent:
            self.setParent(parent)

        self.mpl_connect("pick_event", self.on_pick)
        self.mpl_connect("motion_notify_event", self.on_hover)


    def clear(self):
        self.ax.clear()
        if not hasattr(self.ax, 'get_proj_type'):
            self.ax.get_proj_type = lambda: 'ortho' if getattr(self.ax, '_focal_length', 1) == np.inf else 'persp'
        self._apply_dark_mpl_style_3d()

    def _apply_dark_mpl_style_3d(self):
        fig_bg = "#1f1f1f"
        ax_bg = "#242424"
        text_color = "#e6e6e6"
        spine_color = "#6a6a6a"
        grid_color = "#4a4a4a"

        self.figure.patch.set_facecolor(fig_bg)
        self.ax.set_facecolor(ax_bg)
        self.ax.title.set_color(text_color)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        self.ax.zaxis.label.set_color(text_color)
        self.ax.tick_params(colors=text_color, which="both")

        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis._axinfo["grid"]["color"] = grid_color
            axis._axinfo["grid"]["linewidth"] = 0.6
            axis._axinfo["axisline"]["color"] = spine_color
            axis._axinfo["tick"]["color"] = text_color
            try:
                axis.set_pane_color(mpl.colors.to_rgba(ax_bg, 1.0))
            except Exception:
                pass

    def _apply_light_mpl_style_3d(self):
        fig_bg = "white"
        ax_bg = "white"
        text_color = "black"
        spine_color = "black"
        grid_color = "lightgray"

        self.figure.patch.set_facecolor(fig_bg)
        self.ax.set_facecolor(ax_bg)
        self.ax.title.set_color(text_color)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        self.ax.zaxis.label.set_color(text_color)
        self.ax.tick_params(colors=text_color, which="both")

        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis._axinfo["grid"]["color"] = grid_color
            axis._axinfo["grid"]["linewidth"] = 0.6
            axis._axinfo["axisline"]["color"] = spine_color
            axis._axinfo["tick"]["color"] = text_color
            try:
                axis.set_pane_color(mpl.colors.to_rgba(ax_bg, 1.0))
            except Exception:
                pass

        legend = self.ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("black")
            frame.set_alpha(1.0)
            for text in legend.get_texts():
                text.set_color("black")

        if self.cbar is not None:
            try:
                self.cbar.ax.yaxis.label.set_color("black")
                self.cbar.ax.tick_params(colors="black")
                self.cbar.outline.set_edgecolor("black")
                self.cbar.ax.set_facecolor("white")
            except Exception:
                pass

    def export_projections_white_bg(self, filepath):
        base, ext = os.path.splitext(filepath)
        if not ext:
            ext = ".png"
            filepath = f"{base}{ext}"

        export_dir = os.path.dirname(filepath)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)

        show_tracks = getattr(self.app_instance, 'toggle_tracks', None) is not None and self.app_instance.toggle_tracks.isChecked()
        if not show_tracks and self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        self._apply_light_mpl_style_3d()
        legend = self.ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("black")

        orig_elev = self.ax.elev
        orig_azim = self.ax.azim
        orig_proj = self.ax.get_proj_type()
        orig_aspect = self.ax.get_aspect()
        orig_box_aspect = self.ax.get_box_aspect()
        orig_size = self.figure.get_size_inches()
        orig_xlim = self.ax.get_xlim()
        orig_ylim = self.ax.get_ylim()
        orig_zlim = self.ax.get_zlim()

        # Target uniform square dimensions (1990x1990 at 300 DPI)
        target_size_inches = 1990.0 / 300.0
        self.figure.set_size_inches(target_size_inches, target_size_inches)
        self.ax.set_aspect('equal')
        self.ax.set_box_aspect((1, 1, 1))

        circle_artists = []
        try:
            # Draw Scale Circles at R_200c and 2*R_200c
            if hasattr(self, 'group_r_crit200') and float(getattr(self, 'group_r_crit200', 0.0)) > 0:
                r_val = float(self.group_r_crit200)
                max_r = 2.0 * r_val

                # Ensure view limits accommodate the 2*R_200c virial boundary during export
                cur_max = max(abs(orig_xlim[0]), abs(orig_xlim[1]),
                              abs(orig_ylim[0]), abs(orig_ylim[1]),
                              abs(orig_zlim[0]), abs(orig_zlim[1]))
                export_limit = max(cur_max, max_r * 1.05)
                self.ax.set_xlim([-export_limit, export_limit])
                self.ax.set_ylim([-export_limit, export_limit])
                self.ax.set_zlim([-export_limit, export_limit])

                theta = np.linspace(0, 2 * np.pi, 100)
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                zeros = np.zeros_like(theta)
                for r in [r_val, max_r]:
                    # XY plane (z = 0)
                    lines_xy = self.ax.plot(r * cos_t, r * sin_t, zeros, color='black', linestyle='--', linewidth=1.2, zorder=5)
                    circle_artists.extend(lines_xy)
                    # XZ plane (y = 0)
                    lines_xz = self.ax.plot(r * cos_t, zeros, r * sin_t, color='black', linestyle='--', linewidth=1.2, zorder=5)
                    circle_artists.extend(lines_xz)

            # Export 1 (Perspective)
            self.figure.savefig(filepath, facecolor='white', edgecolor='none', dpi=300, bbox_inches='tight')

            # Export 2 (X-Y Top-Down)
            self.ax.set_proj_type('ortho')
            self.ax.set_aspect('equal')
            self.ax.set_box_aspect((1, 1, 1))
            self.ax.view_init(elev=90, azim=-90)

            # Eliminate Orthogonal Z-Axis Bleed
            self.ax.zaxis.set_ticklabels([])
            self.ax.set_zlabel("")
            self.ax.zaxis.line.set_color((1, 1, 1, 0))

            self.figure.savefig(f"{base}_XY_TopDown{ext}", facecolor='white', edgecolor='none', dpi=300, bbox_inches='tight')

            # Export 3 (X-Z Side-On)
            self.ax.set_zlabel('Z [cKpc/h]')
            self.ax.zaxis.set_major_locator(mpl.ticker.AutoLocator())
            self.ax.zaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            self.ax.zaxis.line.set_color('black')
            self.ax.zaxis.label.set_color('black')
            self.ax.tick_params(colors='black', which='both')

            self.ax.view_init(elev=0, azim=-90)
            self.ax.set_proj_type('ortho')
            self.ax.set_aspect('equal')
            self.ax.set_box_aspect((1, 1, 1))

            # Eliminate Orthogonal Y-Axis Bleed
            self.ax.yaxis.set_ticklabels([])
            self.ax.set_ylabel("")
            self.ax.yaxis.line.set_color((1, 1, 1, 0))

            self.figure.savefig(f"{base}_XZ_SideOn{ext}", facecolor='white', edgecolor='none', dpi=300, bbox_inches='tight')

        finally:
            # Cleanup scale circles
            for artist in circle_artists:
                try:
                    artist.remove()
                except Exception:
                    pass

            # Restore camera, dimensions & limits
            self.ax.view_init(elev=orig_elev, azim=orig_azim)
            self.ax.set_proj_type(orig_proj)
            self.ax.set_aspect(orig_aspect)
            if orig_box_aspect is not None:
                try:
                    self.ax.set_box_aspect(orig_box_aspect)
                except Exception:
                    pass
            self.figure.set_size_inches(orig_size)
            self.ax.set_xlim(orig_xlim)
            self.ax.set_ylim(orig_ylim)
            self.ax.set_zlim(orig_zlim)

            # Restore axis labels, lines, and ticks
            self.ax.set_xlabel('X [cKpc/h]')
            self.ax.set_ylabel('Y [cKpc/h]')
            self.ax.set_zlabel('Z [cKpc/h]')
            for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
                axis.set_major_locator(mpl.ticker.AutoLocator())
                axis.set_major_formatter(mpl.ticker.ScalarFormatter())

            # Restore dark mode
            self._apply_dark_mpl_style_3d()
            legend = self.ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor("#2b2b2b")
                frame.set_edgecolor("#555555")
                frame.set_alpha(0.85)
                for text in legend.get_texts():
                    text.set_color("#e6e6e6")

            # Restore / recreate colorbar if tracks enabled
            if show_tracks:
                if self.cbar is None and getattr(self, 'sm', None) is not None:
                    try:
                        self.cbar = self.figure.colorbar(self.sm, ax=self.ax, fraction=0.03, pad=0.1)
                        self.cbar.set_label("Present  ←  Earlier", rotation=270, labelpad=15)
                    except Exception:
                        pass
                if self.cbar is not None:
                    try:
                        self.cbar.ax.yaxis.label.set_color("#e6e6e6")
                        self.cbar.ax.tick_params(colors="#e6e6e6")
                        self.cbar.outline.set_edgecolor("#555555")
                        self.cbar.ax.set_facecolor("#242424")
                    except Exception:
                        pass
            else:
                if self.cbar is not None:
                    try:
                        self.cbar.remove()
                    except Exception:
                        pass
                    self.cbar = None

            self.draw()

    def export_white_bg(self, filepath):
        self.export_projections_white_bg(filepath)


    def plot_from_files(self, files, central_mass, show_dark=True, show_tracks=True, on_focus=None, group_r_crit200=0.0):
        self.ax.clear()
        if not hasattr(self.ax, 'get_proj_type'):
            self.ax.get_proj_type = lambda: 'ortho' if getattr(self.ax, '_focal_length', 1) == np.inf else 'persp'
        self._apply_dark_mpl_style_3d()
        self.track_data.clear()
        self.dark_subhalo_points.clear()
        self.group_r_crit200 = group_r_crit200

        cmap = plt.get_cmap('cividis')
        sm = None
        track_id = 0
        LBox = None
        central_abs_pos = np.zeros(3)  # Absolute central position read from file

        # Data storage for dynamic limits
        all_x, all_y, all_z = [0.0,], [0.0,], [0.0,]
        min_proximity_distance = float('inf')


        for file in files:
            name = os.path.basename(file).lower()
            try:
                df = pd.read_csv(file)
            except Exception as e:
                print(f"[Error] Could not read {file}: {e}")
                continue

            if 'central_ids' in name:
                try:
                    if 'BoxSize' in df.columns and 'CentralPos_x' in df.columns:
                        # LBox and absolute central position are stored here
                        LBox = df['BoxSize'].dropna().iloc[0]
                        central_abs_pos = np.array([
                            df['CentralPos_x'].iloc[-1],
                            df['CentralPos_y'].iloc[-1],
                            df['CentralPos_z'].iloc[-1]
                        ])


                        if LBox > 0:
                            # Distance to low boundary (0)
                            low_prox_x = central_abs_pos[0]
                            low_prox_y = central_abs_pos[1]
                            low_prox_z = central_abs_pos[2]

                            # Distance to high boundary (LBox)
                            high_prox_x = LBox - central_abs_pos[0]
                            high_prox_y = LBox - central_abs_pos[1]
                            high_prox_z = LBox - central_abs_pos[2]

                            # Find the minimum of all 6 distances
                            min_proximity_distance = min(
                                low_prox_x, low_prox_y, low_prox_z,
                                high_prox_x, high_prox_y, high_prox_z
                            )

                except Exception:
                    pass

            # Plot gradient track lines
            elif 'all_tracks' in name and show_tracks and {'x', 'y', 'z', 'subhalo_index'}.issubset(df.columns):
                if df.empty: continue

                for sub_id in df['subhalo_index'].unique():
                    sub_df = df[df['subhalo_index'] == sub_id][['x', 'y', 'z']].values
                    if len(sub_df) > 1:
                        all_x.extend(sub_df[:, 0])
                        all_y.extend(sub_df[:, 1])
                        all_z.extend(sub_df[:, 2])

                        segments = np.array([sub_df[i:i+2] for i in range(len(sub_df)-1)])
                        norm = mpl.colors.Normalize(vmin=0, vmax=len(sub_df))
                        colors = cmap(norm(np.arange(len(sub_df)-1)))
                        lc = Line3DCollection(segments, colors=colors, linewidth=1.5, picker=True)
                        lc._track_id = track_id
                        self.ax.add_collection3d(lc)

                        # Determine original filename for positions if top 10
                        if "_top_10" in name:
                             base_part = name.split("_all_tracks")[0]
                             pos_file_guess = file.replace(name, base_part + "_positions_top_10.csv")
                        else:
                             base_part = name.split("_all_tracks")[0]
                             pos_file_guess = file.replace(name, base_part + "_positions.csv")

                        self.track_data[track_id] = {
                            "positions": sub_df,
                            "color": "cividis",
                            "subhalo_pos": sub_df[-1],
                            "subhalo_index": int(sub_id),
                            "subhalo_file": pos_file_guess
                        }

                        track_id += 1
                        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

                # Ensure only one pick_event connection
                if on_focus:
                    if self.pick_event_cid is not None:
                        self.mpl_disconnect(self.pick_event_cid)
                    self.pick_event_cid = self.figure.canvas.mpl_connect(
                        'pick_event', lambda event: on_focus(event.artist._track_id)
                    )

            # Plot static positions
            elif 'positions' in name and {'x', 'y', 'z', 'SubhaloMass'}.issubset(df.columns):
                coords = df[['x', 'y', 'z']].values
                masses = df['SubhaloMass'].values
                is_dark = 'dark' in name

                # Check for visibility toggles
                if is_dark and not show_dark:
                    continue

                label = 'Visible Subhalos' if not is_dark else 'Dark Subhalos'
                size = 1 if is_dark else 3

                all_x.extend(coords[:, 0])
                all_y.extend(coords[:, 1])
                all_z.extend(coords[:, 2])

                # Identify the 2nd brightest subhalo for highlighting
                second_brightest_idx = -1
                if not is_dark and len(df) >= 2:
                    # Argsort gives indices that sort the array [min...max].
                    # We want the second to last index for the 2nd heaviest/brightest.
                    sorted_indices = np.argsort(masses)
                    second_brightest_idx = sorted_indices[-2]

                if not is_dark:
                    for i, pos in enumerate(coords):
                        # Skip central (0,0,0) in subhalo plot if strictly overlapping,
                        # but usually subhalos are distinct.
                        
                        # Default Color
                        pt_color = 'green'
                        
                        # Highlight Logic:
                        # 1. If Mass >= Central Mass -> Cyan (Anomaly/Bright System)
                        # 2. If 2nd Brightest Subhalo -> Red (Distinct Highlight)
                        # 3. Else -> Green
                        
                        if masses[i] >= central_mass:
                            pt_color = 'cyan'
                        elif i == second_brightest_idx:
                            pt_color = 'red'
                        
                        # Only label the first point to prevent legend duplication
                        lbl = label if i == 0 else ""
                        self.ax.scatter(pos[0], pos[1], pos[2], color=pt_color, label=lbl, s=size)

                else:
                    self.dark_subhalos_cache = [] # Clear cache before re-population
                    for i, pos in enumerate(coords):
                        self.ax.scatter(pos[0], pos[1], pos[2], color='#aaaaaa', s=size)
                        self.dark_subhalo_points.append({
                            "pos": pos,
                            "subhalo_file": file,
                            "subhalo_index": i
                        })

                    self.dark_subhalos_cache = list(self.dark_subhalo_points)

                    # Create empty hover info box for dark subhalos
                    self.dark_info_text = self.ax.text2D(
                        -0.5, 1.1, "",
                        transform=self.ax.transAxes,
                        fontsize=8,
                        va='top', ha='left',
                        color='#e6e6e6',
                        bbox=dict(facecolor='#1f1f1f', edgecolor='#555555', alpha=0.85)
                    )
                    self.dark_info_text.set_visible(False)

            elif 'metrics' in name:
                continue

            else:
                pass

        # Set central position and plot it
        self.central_pos = np.array([0, 0, 0])
        self.central_mass = central_mass
        self.ax.scatter(*self.central_pos, color='#FFA500', s=self.central_mass, zorder=10, label='Central Subhalo')

        # Toggle simulation box plot checkbox
        if LBox is not None and LBox > 0:
            self.app_instance.toggle_box.show()
        else:
            self.app_instance.toggle_box.hide()

        # Plot limit
        max_coord = 0.0
        if all_x: max_coord = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)), np.max(np.abs(all_z)))
        plot_limit = max(10.0, max_coord * 1.1) # Default zoom

        # Override Limit if Box is Checked
        if LBox is not None and LBox > 0 and self.app_instance.toggle_box.isChecked():
            plot_limit = LBox / 2.0

        self.ax.set_xlim([-plot_limit, plot_limit])
        self.ax.set_ylim([-plot_limit, plot_limit])
        self.ax.set_zlim([-plot_limit, plot_limit])
        self.ax.set_xlabel('X [cKpc/h]')
        self.ax.set_ylabel('Y [cKpc/h]')
        self.ax.set_zlabel('Z [cKpc/h]')

        title_base = f"Subhalos relative to central (LBox: {LBox:.2f})" if LBox else "Subhalo Tracks and Positions"
        if min_proximity_distance != float('inf'):
            title_base += f" | Nearest Boundary: {min_proximity_distance:.2f}"


        self.ax.set_title(title_base)

        # --- Plotting Box Volume and Edges when Toggled ---
        if LBox is not None and LBox > 0 and self.app_instance.toggle_box.isChecked():

            min_rel = 0 - central_abs_pos
            max_rel = LBox - central_abs_pos

            # Generate mesh grid for plotting surfaces
            Y_range = np.linspace(min_rel[1], max_rel[1], 2)
            Z_range = np.linspace(min_rel[2], max_rel[2], 2)
            X_range = np.linspace(min_rel[0], max_rel[0], 2)

            # X-Y faces (Z-constant faces)
            X, Y = np.meshgrid(X_range, Y_range)
            X_yz, Z_yz = np.meshgrid(X_range, Z_range)
            Y_xz, Z_xz = np.meshgrid(Y_range, Z_range)

            # Define vertices for edges plotting
            corners = [0, LBox]
            vertices = np.array([
                [x - central_abs_pos[0], y - central_abs_pos[1], z - central_abs_pos[2]]
                for x in corners for y in corners for z in corners
            ])
            edges = [
                [0, 1], [1, 3], [3, 2], [2, 0],
                [4, 5], [5, 7], [7, 6], [6, 4],
                [0, 4], [1, 5], [3, 7], [2, 6]
            ]

            # 1. Plot the 6 transparent faces (Volume)
            color_face = 'red'
            alpha_face = 0.05
            self.ax.plot_surface(X, Y, Z=min_rel[2] * np.ones_like(X), color=color_face, alpha=alpha_face, zorder=0)
            self.ax.plot_surface(X, Y, Z=max_rel[2] * np.ones_like(X), color=color_face, alpha=alpha_face, zorder=0)
            self.ax.plot_surface(X=X_yz, Y=min_rel[1] * np.ones_like(X_yz), Z=Z_yz, color=color_face, alpha=alpha_face, zorder=0)
            self.ax.plot_surface(X=X_yz, Y=max_rel[1] * np.ones_like(X_yz), Z=Z_yz, color=color_face, alpha=alpha_face, zorder=0)
            self.ax.plot_surface(X=min_rel[0] * np.ones_like(Y_xz), Y=Y_xz, Z=Z_xz, color=color_face, alpha=alpha_face, zorder=0)
            self.ax.plot_surface(X=max_rel[0] * np.ones_like(Y_xz), Y=Y_xz, Z=Z_xz, color=color_face, alpha=alpha_face, zorder=0)

            # 2. Plot the 12 wireframe edges
            color_edge = 'gray'
            for edge in edges:
                points = vertices[edge]
                self.ax.plot(points[:, 0], points[:, 1], points[:, 2],
                             color=color_edge, linestyle='-', linewidth=0.7, alpha=0.8, zorder=1) # Zorder 1 ensures edges are slightly above the surface

        legend_handles, legend_labels = self.ax.get_legend_handles_labels()

        # Update legend to include special highlight colors
        if not is_dark:
            # Check if we actually plotted these colors to avoid empty legend entries
            # We assume if the logic ran, they might exist. 
            legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='2nd Brightest'))
            legend_labels.append('2nd Brightest')
            legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=6, label='Brighter than Central'))
            legend_labels.append('Brighter than Central')

        # If the box is plotted, add box labels
        if self.app_instance.toggle_box.isChecked():
            if LBox is not None and LBox > 0:
                legend_handles.append(Line2D([0], [0], color='gray', lw=0.5, ls='-', label='Simulation Box'))
                legend_labels.append('Simulation Box')

        # De-duplicate legend entries
        unique_handles = []
        unique_labels = []
        seen = set()
        for h, l in zip(legend_handles, legend_labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)

        self.ax.legend(handles=unique_handles, labels=unique_labels, fontsize='small')
        legend = self.ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor("#2b2b2b")
            frame.set_edgecolor("#555555")
            frame.set_alpha(0.85)
            for text in legend.get_texts():
                text.set_color("#e6e6e6")

        if sm:
            self.sm = sm
            if self.cbar is None:
                self.cbar = self.figure.colorbar(sm, ax=self.ax, fraction=0.03, pad=0.1)
                self.cbar.set_label("Present  ←  Earlier", rotation=270, labelpad=15)
            else:
                self.cbar.update_normal(sm)
            self.cbar.ax.yaxis.label.set_color("#e6e6e6")
            self.cbar.ax.tick_params(colors="#e6e6e6")
            self.cbar.outline.set_edgecolor("#555555")
            self.cbar.ax.set_facecolor("#242424")
        else:
            if self.cbar is not None:
                try:
                    self.cbar.remove()
                except Exception:
                    pass
                self.cbar = None

        self.draw()


    def on_pick(self, event):
        artist = event.artist
        if hasattr(artist, "_track_id"):
            track_id = artist._track_id
            if hasattr(self.app_instance, 'focus_on_track'):
                self.app_instance.focus_on_track(track_id)


    def show_single_track(self, track_id):
        self.ax.clear()
        self._apply_dark_mpl_style_3d()
        self.dark_subhalo_points.clear()  # Reset for fresh hover tracking

        # Central subhalo
        self.ax.scatter(*self.central_pos, color='#FFA500', s=self.central_mass)
        legend_handles, legend_labels = self.ax.get_legend_handles_labels()
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
            markerfacecolor='#FFA500', markersize=6, label='Central Subhalo'))
        legend_labels.append('Central Subhalo')

        # Plot all tracks (faded except selected)
        cmap = plt.get_cmap('cividis')
        for tid, data in self.track_data.items():
            positions = data["positions"]
            norm = mpl.colors.Normalize(vmin=0, vmax=len(positions))
            colors = cmap(norm(np.arange(len(positions)-1)))
            alpha_val = 0.2 if tid != track_id else 1.0
            linewidth_val = 0.8 if tid != track_id else 2.5
            lc = Line3DCollection(
                np.array([positions[i:i+2] for i in range(len(positions)-1)]),
                colors=colors, linewidth=linewidth_val, alpha=alpha_val, picker=True
            )
            lc._track_id = tid
            self.ax.add_collection3d(lc)

        # Highlight the selected subhalo
        subhalo_pos = self.track_data[track_id]["subhalo_pos"]
        self.ax.scatter(*subhalo_pos, color='green', s=30, label='Selected Subhalo')

        if self.app_instance.toggle_dark.isChecked() and hasattr(self, "dark_subhalos_cache"):
            for info in self.dark_subhalos_cache:
                self.ax.scatter(info["pos"][0], info["pos"][1], info["pos"][2],
                                color='#aaaaaa', s=1) # No label here to avoid duplication in legend
                self.dark_subhalo_points.append(info)

            # Add Dark Subhalos to the legend explicitly (if the label isn't already there)
            legend_handles, legend_labels = self.ax.get_legend_handles_labels()
            if 'Dark Subhalos' not in legend_labels:
                 legend_handles.append(Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='#aaaaaa', markersize=3, label='Dark Subhalos'))
                 legend_labels.append('Dark Subhalos')

        # Recreate hover info box (empty initially)
        self.dark_info_text = self.ax.text2D(
            -0.5, 1.1, "",
            transform=self.ax.transAxes,
            fontsize=8,
            va='top', ha='left',
            color='#e6e6e6',
            bbox=dict(facecolor='#1f1f1f', edgecolor='#555555', alpha=0.85)
        )
        self.dark_info_text.set_visible(False)

        # Load properties of the focused track’s subhalo
        props_msg = "No properties found."
        subhalo_index = self.track_data[track_id].get("subhalo_index", None)
        subhalo_file = self.track_data[track_id].get("subhalo_file", None)
        if subhalo_file and os.path.exists(subhalo_file) and subhalo_index is not None:
            try:
                df = pd.read_csv(subhalo_file)
                if 0 <= subhalo_index < len(df):
                    props_series = df.iloc[subhalo_index]
                    props_msg = "\n".join(f"{col}: {val}" for col, val in props_series.items())
            except Exception as e:
                props_msg = f"Error reading properties: {e}"

        if hasattr(self.app_instance, "update_subhalo_info"):
            self.app_instance.update_subhalo_info(props_msg)

        self.ax.legend(handles=legend_handles, labels=legend_labels, fontsize='small')
        legend = self.ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor("#2b2b2b")
            frame.set_edgecolor("#555555")
            frame.set_alpha(0.85)
            for text in legend.get_texts():
                text.set_color("#e6e6e6")
        self.ax.set_title(f"Focused View: Track {track_id}")
        self.draw()


    def on_hover(self, event):
        if not event.inaxes or event.inaxes != self.ax:
            if self.dark_info_text is not None:
                self.dark_info_text.set_visible(False)
                self.draw()
            return

        min_dist = float("inf")
        nearest_info = None

        # Project 3D coords to 2D display coords
        for info in self.dark_subhalo_points:
            x2, y2, _ = proj3d.proj_transform(*info["pos"], self.ax.get_proj())
            trans = self.ax.transData.transform((x2, y2))
            dist = np.hypot(trans[0] - event.x, trans[1] - event.y)
            if dist < min_dist:
                min_dist = dist
                nearest_info = info

        if nearest_info and min_dist < 5:
            self.update_dark_info_text(nearest_info)
        else:
            if self.dark_info_text is not None:
                self.dark_info_text.set_visible(False)
                self.draw()


    def update_dark_info_text(self, info):
        props_msg = "Dark Subhalo"
        subhalo_file = info.get("subhalo_file")
        subhalo_index = info.get("subhalo_index")

        if subhalo_file and os.path.exists(subhalo_file) and subhalo_index is not None:
            try:
                df = pd.read_csv(subhalo_file)
                if 0 <= subhalo_index < len(df):
                    props_series = df.iloc[subhalo_index]
                    props_msg = "\n".join(f"{col}: {val}" for col, val in props_series.items())
            except Exception as e:
                props_msg = f"Error reading properties: {e}"

        if self.dark_info_text is None:
            self.dark_info_text = self.ax.text2D(
                -0.5, 1.1, props_msg,
                transform=self.ax.transAxes,
                fontsize=8,
                va='top', ha='left',
                color='black',
                bbox=dict(facecolor='white', alpha=0.8)
            )
        else:
            self.dark_info_text.set_text(props_msg)

        self.dark_info_text.set_visible(True)
        self.draw()


# --- Utility Dictionaries for Plotting ---
CATEGORY_COLORS = {
    'Most Lopsided (Top 1/3)': 'red',
    'Least Lopsided (Bottom 1/3)': 'blue',
    'Tracked': 'gray'
}


# --- Canvas for Metric Evolution (Single Halo) ---
class MetricEvolutionCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.subplots(3, 2)
        super(MetricEvolutionCanvas, self).__init__(self.figure)
        self.setParent(parent)
        self.current_data = None
        self.current_title_suffix = ""
        self._apply_dark_mpl_style_2d()

    def _apply_dark_mpl_style_2d(self):
        fig_bg = "#1f1f1f"
        ax_bg = "#242424"
        text_color = "#e6e6e6"
        spine_color = "#5a5a5a"
        grid_color = "#4a4a4a"

        self.figure.patch.set_facecolor(fig_bg)
        self.figure.set_facecolor(fig_bg)
        for ax in self.axes.flatten():
            ax.set_facecolor(ax_bg)
            ax.title.set_color(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(colors=text_color, which="both")
            ax.grid(True, linestyle="--", alpha=0.55, color=grid_color)
            for spine in ax.spines.values():
                spine.set_color(spine_color)
            legend = ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor("#2b2b2b")
                frame.set_edgecolor("#555555")
                frame.set_alpha(0.85)
                for text in legend.get_texts():
                    text.set_color("#e6e6e6")
        if hasattr(self.figure, "_suptitle") and self.figure._suptitle is not None:
            self.figure._suptitle.set_color(text_color)
        if hasattr(self.figure, "_supxlabel") and self.figure._supxlabel is not None:
            self.figure._supxlabel.set_color(text_color)

    def _apply_light_mpl_style_2d(self):
        fig_bg = "white"
        ax_bg = "white"
        text_color = "black"
        spine_color = "black"
        grid_color = "lightgray"

        self.figure.patch.set_facecolor(fig_bg)
        self.figure.set_facecolor(fig_bg)
        for ax in self.axes.flatten():
            ax.set_facecolor(ax_bg)
            ax.title.set_color(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(colors=text_color, which="both")
            ax.grid(True, linestyle="--", alpha=0.55, color=grid_color)
            for spine in ax.spines.values():
                spine.set_color(spine_color)
            legend = ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor("white")
                frame.set_edgecolor("black")
                frame.set_alpha(1.0)
                for text in legend.get_texts():
                    text.set_color("black")
        if hasattr(self.figure, "_suptitle") and self.figure._suptitle is not None:
            self.figure._suptitle.set_color(text_color)
        if hasattr(self.figure, "_supxlabel") and self.figure._supxlabel is not None:
            self.figure._supxlabel.set_color(text_color)

    def plot_evolution(self, data, title_suffix=""):
        self.current_data = data
        self.current_title_suffix = title_suffix
        for ax in self.axes.flatten():
            ax.clear()
        self._apply_dark_mpl_style_2d()

        if not data or not data.get("SnapNum"):
            self.clear_and_show_message("No metric evolution data to display.")
            return

        df = pd.DataFrame(data)
        if 'Centroid_Offset' not in df.columns and 'Centroid_Distance' in df.columns:
            df.rename(columns={'Centroid_Distance': 'Centroid_Offset'}, inplace=True)

        if df.empty or 'SnapNum' not in df.columns or len(df['SnapNum']) == 0:
            self.clear_and_show_message("Metric evolution data is empty or invalid.")
            return

        metrics_list = [
            ('Centroid_Offset', 'Centroid Offset', 'blue', (0, 0)),
            ('MRL', 'MRL', 'green', (0, 1)),
            ('Pairwise', 'Pairwise Angle', 'red', (1, 0)),
            ('Hemi_Contrast', 'Hemi Contrast', 'purple', (1, 1)),
            ('Normalised_RMS_Distance', 'Norm. RMS Distance', 'orange', (2, 0)),
        ]

        min_snap = df['SnapNum'].min()
        max_snap = df['SnapNum'].max()

        for metric_key, label, color, pos in metrics_list:
            if metric_key in df.columns:
                ax = self.axes[pos]
                ax.plot(df['SnapNum'], df[metric_key], marker='o', linestyle='-', color=color)
                ax.set_ylabel(label)
                ax.tick_params(axis='x', rotation=45)
                if pd.notna(min_snap) and pd.notna(max_snap):
                    ax.set_xlim(min_snap, max_snap)

        self.axes[2, 1].axis('off') # Turn off the unused 6th plot

        self.figure.suptitle(f'Metric Evolution {title_suffix}', fontsize=16)
        self.figure.supxlabel('Snapshot Number')
        self.figure._suptitle.set_color("#e6e6e6")
        if self.figure._supxlabel is not None:
            self.figure._supxlabel.set_color("#e6e6e6")
        self.figure.tight_layout(pad=1.0, h_pad=0.5, rect=[0, 0, 1, 0.96])
        self.draw()

    def clear_and_show_message(self, message):
        self.current_data = None
        self.current_title_suffix = ""
        for ax in self.axes.flatten():
            ax.clear()
        self._apply_dark_mpl_style_2d()
        self.figure.suptitle(message, fontsize=12)
        self.figure._suptitle.set_color("#e6e6e6")
        if self.axes.shape == (3, 2):
            self.axes[2, 1].axis('off')
        self.draw()

    def export_all_light(self, filepath):
        self._apply_light_mpl_style_2d()
        try:
            self.figure.savefig(filepath, facecolor='white', dpi=300, bbox_inches='tight')
        finally:
            self._apply_dark_mpl_style_2d()
            self.draw()

    def export_core_light(self, filepath):
        if not hasattr(self, 'current_data') or not self.current_data:
            return

        df = pd.DataFrame(self.current_data)
        if 'Centroid_Offset' not in df.columns and 'Centroid_Distance' in df.columns:
            df.rename(columns={'Centroid_Distance': 'Centroid_Offset'}, inplace=True)

        if df.empty or 'SnapNum' not in df.columns or len(df['SnapNum']) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor("white")
        fig.set_facecolor("white")

        metrics_list = [
            ('Centroid_Offset', 'Centroid Offset', 'blue', (0, 0)),
            ('MRL', 'MRL', 'green', (0, 1)),
            ('Pairwise', 'Pairwise Angle', 'red', (1, 0)),
            ('Hemi_Contrast', 'Hemi Contrast', 'purple', (1, 1)),
        ]

        min_snap = df['SnapNum'].min()
        max_snap = df['SnapNum'].max()

        for metric_key, label, color, pos in metrics_list:
            ax = axes[pos]
            ax.set_facecolor("white")
            ax.title.set_color("black")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.tick_params(colors="black", which="both")
            ax.grid(True, linestyle="--", alpha=0.55, color="lightgray")
            for spine in ax.spines.values():
                spine.set_color("black")

            if metric_key in df.columns:
                ax.plot(df['SnapNum'], df[metric_key], marker='o', linestyle='-', color=color)
                ax.set_ylabel(label)
                ax.tick_params(axis='x', rotation=45)
                if pd.notna(min_snap) and pd.notna(max_snap):
                    ax.set_xlim(min_snap, max_snap)

        suffix = getattr(self, "current_title_suffix", "")
        fig.suptitle(f'Core Metric Evolution {suffix}', fontsize=16, color="black")
        fig.supxlabel('Snapshot Number', color="black")
        fig.tight_layout(pad=1.0, h_pad=0.5, rect=[0, 0, 1, 0.96])
        try:
            fig.savefig(filepath, facecolor='white', dpi=300, bbox_inches='tight')
        finally:
            plt.close(fig)


# --- Canvas for Metric Overview (All Halos) ---
class MetricOverviewCanvas(MetricEvolutionCanvas):
    def export_overview_white_bg(self, filepath):
        self.export_all_light(filepath)

    def plot_overview(self, aggregated_data, mode="Satellite-Centric"):
        for ax in self.axes.flatten():
            ax.clear()
        if self.axes.shape == (3, 2):
            self.axes[2, 1].axis('off')
        self._apply_dark_mpl_style_2d()

        # Check if we actually have data to plot
        has_data = False
        all_snaps = []
        for category, data in aggregated_data.items():
            if data.get("SnapNum") and len(data["SnapNum"]) > 0:
                has_data = True
                all_snaps.extend(data["SnapNum"])

        if not aggregated_data or not has_data:
            self.clear_and_show_message(f"No {mode} overview data found.\n(Check if *_metric_evolution.csv files exist for Satellite-Centric mode)")
            return

        min_snap = min(all_snaps) if all_snaps else None
        max_snap = max(all_snaps) if all_snaps else None

        metrics_list = [
            ('Centroid_Offset', 'Centroid Offset', (0, 0)),
            ('MRL', 'MRL', (0, 1)),
            ('Pairwise', 'Pairwise Angle', (1, 0)),
            ('Hemi_Contrast', 'Hemi Contrast', (1, 1)),
            ('Normalised_RMS_Distance', 'Norm. RMS Distance', (2, 0)),
        ]

        # Plot data for each category (now implicitly 'Tracked')
        for category, data in aggregated_data.items():
            if not data.get("SnapNum"): continue

            if 'Centroid_Distance' in data and 'Centroid_Offset' not in data:
                data['Centroid_Offset'] = data.pop('Centroid_Distance')

            snap_nums = data['SnapNum']
            color = CATEGORY_COLORS.get(category, 'gray')

            for metric_key, label, pos in metrics_list:
                if metric_key in data and data[metric_key]:
                    ax = self.axes[pos]
                    stats = compute_percentile_band(
                        data[metric_key],
                        lower_percentile=16,
                        upper_percentile=84,
                    )
                    median = stats["median"]
                    low_bound = stats["low_bound"]
                    high_bound = stats["high_bound"]

                    # Plot the Median line
                    ax.plot(snap_nums, median,
                            linestyle='-', color=color, linewidth=2, label=f'Tracked Median', zorder=1)

                    # Plot the 16th and 84th percentiles
                    ax.fill_between(snap_nums, low_bound, high_bound,
                                    color=color, alpha=0.4, linewidth=0, zorder=1)

                    ax.set_ylabel(label)
                    ax.tick_params(axis='x', rotation=45)

        # Final adjustments
        for metric_key, label, pos in metrics_list:
            ax = self.axes[pos]
            if not ax.get_legend_handles_labels()[0]:
                ax.set_ylabel(label)
            if min_snap is not None and max_snap is not None:
                ax.set_xlim(min_snap, max_snap)

        # Add legend ONLY if there are handles (prevents warning if no data plotted)
        handles, labels = self.axes[0, 0].get_legend_handles_labels()
        if handles:
            self.axes[0, 0].legend(fontsize='small')
            legend = self.axes[0, 0].get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor("#2b2b2b")
                frame.set_edgecolor("#555555")
                frame.set_alpha(0.85)
                for text in legend.get_texts():
                    text.set_color("#e6e6e6")

        self.figure.suptitle(f'Metrics Overview: {mode} Evolution (Median & $\\pm 1\\sigma$ of ALL Tracked Halos)', fontsize=16)
        self.figure.supxlabel('Snapshot Number')
        self.figure._suptitle.set_color("#e6e6e6")
        if self.figure._supxlabel is not None:
            self.figure._supxlabel.set_color("#e6e6e6")
        self.figure.tight_layout(pad=1.0, h_pad=0.5, rect=[0, 0, 1, 0.96])
        self.draw()



# ---------------------------------------------------------------------------
# Statistical Significance — Canvas (2×2 Matplotlib grid)
# ---------------------------------------------------------------------------

class PValueCanvas(FigureCanvas):
    """
    A 2×2 Matplotlib canvas that renders either:

    * **Single Halo mode** — null-distribution histograms with vertical lines
      marking the observed value for each of the 4 core metrics.
    * **Tracked Halo History mode** — temporal p-value trajectories across
      snapshots for each of the 4 core metrics.
    """

    # Panel layout: (row, col) → (metric_key, display_label, plot_color)
    PANELS = [
        ((0, 0), "MRL",             "Mean Resultant Length", "#4caf50"),
        ((0, 1), "Pairwise",        "Pairwise Angle (°)",    "#ef5350"),
        ((1, 0), "Hemi_Contrast",   "Hemi Contrast",         "#ab47bc"),
        ((1, 1), "Centroid_Offset", "Centroid Offset",       "#42a5f5"),
    ]

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(9, 6), dpi=100)
        self.axes = self.figure.subplots(2, 2)
        super().__init__(self.figure)
        self.setParent(parent)
        self._apply_dark_mpl_style_2d()

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------

    def _apply_dark_mpl_style_2d(self):
        fig_bg     = "#1f1f1f"
        ax_bg      = "#242424"
        text_color = "#e6e6e6"
        spine_color = "#5a5a5a"
        grid_color  = "#4a4a4a"

        self.figure.patch.set_facecolor(fig_bg)
        self.figure.set_facecolor(fig_bg)
        for ax in self.axes.flatten():
            ax.set_facecolor(ax_bg)
            ax.title.set_color(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(colors=text_color, which="both")
            ax.grid(True, linestyle="--", alpha=0.55, color=grid_color)
            for spine in ax.spines.values():
                spine.set_color(spine_color)
        if hasattr(self.figure, "_suptitle") and self.figure._suptitle is not None:
            self.figure._suptitle.set_color(text_color)

    def _apply_light_mpl_style_2d(self):
        fig_bg     = "white"
        ax_bg      = "white"
        text_color = "black"
        spine_color = "black"
        grid_color  = "lightgray"

        self.figure.patch.set_facecolor(fig_bg)
        self.figure.set_facecolor(fig_bg)
        for ax in self.axes.flatten():
            ax.set_facecolor(ax_bg)
            ax.title.set_color(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(colors=text_color, which="both")
            ax.grid(True, linestyle="--", alpha=0.55, color=grid_color)
            for spine in ax.spines.values():
                spine.set_color(spine_color)
        if hasattr(self.figure, "_suptitle") and self.figure._suptitle is not None:
            self.figure._suptitle.set_color("black")

    # ------------------------------------------------------------------
    # Single Halo mode: null-distribution histograms
    # ------------------------------------------------------------------

    def plot_null_distributions(
        self,
        null_dists: dict,
        observed_values: dict,
        p_iso_values: dict,
        p_cosmo_values: dict,
        n_sat: int,
    ):
        """
        Draw null distribution histograms with observed-value markers.

        Parameters
        ----------
        null_dists : dict
            Keys ``"R_bar"``, ``"theta_lop"``, ``"C_hemi"`` → sorted float32 arrays.
            ``"D_cen"`` is optional.
        observed_values : dict
            Keys ``"MRL"``, ``"Pairwise"``, ``"Hemi_Contrast"``, ``"Centroid_Offset"``.
        p_iso_values : dict
            Keys same as *observed_values*.
        p_cosmo_values : dict
            Keys same as *observed_values*.
        n_sat : int
            Satellite multiplicity (for title annotation).
        """
        # Map metric key → null array key
        null_key_map = {
            "MRL":             "R_bar",
            "Pairwise":        "theta_lop",
            "Hemi_Contrast":   "C_hemi",
            "Centroid_Offset": "D_cen",
        }

        for ax in self.axes.flatten():
            ax.clear()
        self._apply_dark_mpl_style_2d()

        for (row, col), metric_key, label, color in self.PANELS:
            ax = self.axes[row, col]
            null_arr = null_dists.get(null_key_map[metric_key])
            obs_val  = observed_values.get(metric_key)
            p_iso    = p_iso_values.get(metric_key)
            p_cosmo  = p_cosmo_values.get(metric_key)

            if null_arr is None or len(null_arr) == 0:
                ax.text(0.5, 0.5, "No null distribution",
                        ha="center", va="center", transform=ax.transAxes,
                        color="#e6e6e6", fontsize=9)
                ax.set_title(label, fontsize=9)
                continue

            ax.hist(null_arr, bins=60, color=color, alpha=0.55,
                    edgecolor="none", density=True)

            if obs_val is not None and obs_val == obs_val:  # not NaN
                ax.axvline(obs_val, color="#ff4444", linestyle="--",
                           linewidth=1.8, zorder=5)
                p_iso_str   = f"{p_iso:.3f}"   if (p_iso   is not None and p_iso   == p_iso)   else "N/A"
                p_cosmo_str = f"{p_cosmo:.3f}" if (p_cosmo is not None and p_cosmo == p_cosmo) else "N/A"
                ax.text(
                    0.97, 0.97,
                    f"$p_{{\\rm iso}}={p_iso_str}$\n$p_{{\\rm cosmo}}={p_cosmo_str}$",
                    ha="right", va="top", transform=ax.transAxes,
                    color="#ff8888", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#2a2a2a", alpha=0.75),
                )

            ax.set_title(label, fontsize=9)
            ax.set_ylabel("Density", fontsize=8)

        self.figure.suptitle(
            f"Isotropic Null Distributions  ($N_{{\\rm sat}}={n_sat}$)",
            fontsize=10,
        )
        self.figure.tight_layout(pad=0.8, rect=[0, 0, 1, 0.95])
        self.draw()

    # ------------------------------------------------------------------
    # Tracked Halo History mode: temporal p-value trajectories
    # ------------------------------------------------------------------

    def plot_temporal_significance(
        self,
        snap_nums: list,
        p_iso_by_metric: dict,
        p_cosmo_by_metric: dict,
    ):
        """
        Draw temporal p-value trajectories (p_iso + p_cosmo) for each metric.

        Parameters
        ----------
        snap_nums : list[int]
            Snapshot numbers in ascending order.
        p_iso_by_metric : dict
            Keys = metric_key → list of floats (one per snapshot, NaN if missing).
        p_cosmo_by_metric : dict
            Same structure.
        """
        for ax in self.axes.flatten():
            ax.clear()
        self._apply_dark_mpl_style_2d()

        for (row, col), metric_key, label, color in self.PANELS:
            ax = self.axes[row, col]
            p_iso   = p_iso_by_metric.get(metric_key, [])
            p_cosmo = p_cosmo_by_metric.get(metric_key, [])

            snaps = np.array(snap_nums, dtype=float)
            p_iso_arr   = np.array(p_iso,   dtype=float)
            p_cosmo_arr = np.array(p_cosmo, dtype=float)

            if len(snaps) > 0:
                ax.plot(snaps, p_iso_arr, color=color, linestyle="--",
                        linewidth=1.5, marker="o", markersize=3,
                        label="$p_{\\rm iso}$", alpha=0.9)
                ax.plot(snaps, p_cosmo_arr, color=color, linestyle="-",
                        linewidth=1.5, marker="s", markersize=3,
                        label="$p_{\\rm cosmo}$", alpha=0.7)
                ax.axhline(0.05, color="#ff4444", linestyle="--",
                           linewidth=1.2, alpha=0.8, label="$p=0.05$")
                ax.set_ylim(-0.02, 1.05)
                ax.legend(fontsize=7, loc="upper left",
                          facecolor="#2b2b2b", edgecolor="#555555",
                          labelcolor="#e6e6e6", framealpha=0.75)

            ax.set_title(label, fontsize=9)
            ax.set_xlabel("Snapshot", fontsize=8)
            ax.set_ylabel("$p$-value", fontsize=8)

        self.figure.suptitle("Temporal Significance Evolution", fontsize=10)
        self.figure.tight_layout(pad=0.8, rect=[0, 0, 1, 0.95])
        self.draw()

    def clear_and_show_message(self, message: str = ""):
        for ax in self.axes.flatten():
            ax.clear()
        self._apply_dark_mpl_style_2d()
        if message:
            self.axes[0, 0].text(
                0.5, 0.5, message,
                ha="center", va="center",
                transform=self.axes[0, 0].transAxes,
                color="#aaaaaa", fontsize=10,
            )
        self.draw()

    def export_white_bg(self, filepath: str):
        """Save the figure to *filepath* with a white background."""
        self._apply_light_mpl_style_2d()
        try:
            self.figure.savefig(filepath, facecolor="white", dpi=300,
                                bbox_inches="tight")
        finally:
            self._apply_dark_mpl_style_2d()
            self.draw()


# ---------------------------------------------------------------------------
# Statistical Significance — Tab Widget
# ---------------------------------------------------------------------------

class PValueTab(QWidget):
    """
    Composite widget shown as the "Statistical Significance (p-values)" tab.

    Contains:
    - A controls row (Recalculate button + status label)
    - A QTableWidget summarising p-values per metric (single-halo mode)
    - A PValueCanvas for null histograms or temporal trajectories
    """

    METRIC_LABELS = [
        ("MRL",             "Mean Resultant Length"),
        ("Pairwise",        "Pairwise Angle"),
        ("Hemi_Contrast",   "Hemi Contrast"),
        ("Centroid_Offset", "Centroid Offset"),
    ]
    # Background colour for p < 0.05 cells (soft dark green)
    SIG_BG    = QColor("#364a36")
    SIG_FG    = QColor("#b0e0b0")
    NOSIG_FG  = QColor("#e6e6e6")
    NOSIG_BG  = QColor("#232323")

    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # ---- Controls row ----
        ctrl_row = QHBoxLayout()
        self.btn_recalculate = QPushButton("\u27f3  Recalculate / Force Re-seed Cache")
        self.btn_recalculate.setMaximumWidth(280)
        self.btn_export_light = QPushButton("Export Plot (Light BG)")
        self.btn_export_light.setMaximumWidth(180)
        self.chk_top10 = QCheckBox("Restrict to Top 10 Satellites")
        self.lbl_status = QLabel("No data loaded")
        self.lbl_status.setStyleSheet("color: #888888; font-style: italic; font-size: 11px;")
        ctrl_row.addWidget(self.btn_recalculate)
        ctrl_row.addWidget(self.btn_export_light)
        ctrl_row.addWidget(self.chk_top10)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self.lbl_status, stretch=1)
        main_layout.addLayout(ctrl_row)

        # ---- Summary table ----
        self.table = QTableWidget(4, 5)
        self.table.setHorizontalHeaderLabels([
            "Metric",
            "Observed Value",
            "p_cosmo",
            "p_iso",
            "Cosmo Rank / Total",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMaximumHeight(170)

        for row, (_, label) in enumerate(self.METRIC_LABELS):
            item = QTableWidgetItem(label)
            item.setForeground(self.NOSIG_FG)
            self.table.setItem(row, 0, item)

        main_layout.addWidget(self.table)

        # ---- Canvas ----
        self.canvas = PValueCanvas()
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(NavigationToolbar(self.canvas, self))

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def show_insufficient_data(self, n_sat: int):
        """Display a message when N_sat < 2 (metrics are undefined)."""
        self._clear_table()
        self.canvas.clear_and_show_message(
            f"N_sat = {n_sat} — angular statistics require ≥ 2 satellites."
        )
        self.lbl_status.setText(f"N/A (N_sat = {n_sat} < 2)")

    def update_for_single_halo(
        self,
        observed_values: dict,
        p_iso_values: dict,
        p_cosmo_values: dict,
        rank_info: dict,
        null_dists: dict,
        n_sat: int,
        from_cache: bool,
    ):
        """
        Refresh table + histogram canvas for Single Halo mode.

        Parameters
        ----------
        observed_values, p_iso_values, p_cosmo_values, rank_info : dicts
        null_dists : dict  — "R_bar", "theta_lop", "C_hemi", optionally "D_cen"
        n_sat : int
        from_cache : bool  — whether distributions came from .p_val_cache
        """
        self._populate_table(observed_values, p_iso_values, p_cosmo_values, rank_info)
        self.canvas.plot_null_distributions(
            null_dists, observed_values, p_iso_values, p_cosmo_values, n_sat
        )
        if from_cache:
            self.lbl_status.setText("✓  Loaded from .p_val_cache")
            self.lbl_status.setStyleSheet("color: #6dbf6d; font-style: normal; font-size: 11px;")
        else:
            self.lbl_status.setText(f"✦  Freshly generated (N = {100_000:,})")
            self.lbl_status.setStyleSheet("color: #e6a020; font-style: normal; font-size: 11px;")

    def update_for_tracked_history(
        self,
        snap_nums: list,
        p_iso_by_metric: dict,
        p_cosmo_by_metric: dict,
    ):
        """
        Refresh temporal p-value canvas for Tracked Halo History mode.
        The summary table is hidden in this mode.
        """
        self.table.setVisible(False)
        self.canvas.plot_temporal_significance(snap_nums, p_iso_by_metric, p_cosmo_by_metric)
        self.lbl_status.setText("Tracked history — temporal significance")
        self.lbl_status.setStyleSheet("color: #888888; font-style: italic; font-size: 11px;")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clear_table(self):
        self.table.setVisible(True)
        for row in range(4):
            for col in range(1, 5):
                item = QTableWidgetItem("—")
                item.setForeground(self.NOSIG_FG)
                item.setBackground(self.NOSIG_BG)
                self.table.setItem(row, col, item)

    def _populate_table(
        self,
        observed_values: dict,
        p_iso_values: dict,
        p_cosmo_values: dict,
        rank_info: dict,
    ):
        self.table.setVisible(True)
        nan = float("nan")

        for row, (metric_key, label) in enumerate(self.METRIC_LABELS):
            obs    = observed_values.get(metric_key, nan)
            p_iso  = p_iso_values.get(metric_key, nan)
            p_cosmo = p_cosmo_values.get(metric_key, nan)
            rank, total = rank_info.get(metric_key, (nan, nan))

            obs_str     = f"{obs:.4f}"      if obs     == obs     else "N/A"
            p_cosmo_str = f"{p_cosmo:.4f}"  if p_cosmo == p_cosmo else "N/A"
            p_iso_str   = f"{p_iso:.4f}"    if p_iso   == p_iso   else "N/A"
            rank_str    = (f"{int(rank)} / {int(total)}"
                           if (rank == rank and total == total) else "N/A")

            # Determine significance (p < 0.05 on either test)
            is_sig = ((p_iso   == p_iso   and p_iso   < 0.05) or
                      (p_cosmo == p_cosmo and p_cosmo < 0.05))
            bg = self.SIG_BG  if is_sig else self.NOSIG_BG
            fg = self.SIG_FG  if is_sig else self.NOSIG_FG

            values = [obs_str, p_cosmo_str, p_iso_str, rank_str]
            for col_offset, val_str in enumerate(values):
                item = QTableWidgetItem(val_str)
                item.setForeground(fg)
                item.setBackground(bg)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col_offset + 1, item)



# --- Satellite-Centric Significance Evolution Tab ---
class SatCentricTab(QWidget):
    """
    Tab showing satellite-centric temporal p-value evolution for a single halo.

    Displays p_iso(t) and p_cosmo(t) across Snapshots 68→99 for the four
    lopsidedness metrics, using the fixed progenitor set (Satellite-Centric
    frame).  Visible only in Single Halo View mode.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # ---- Controls row ----
        ctrl_row = QHBoxLayout()
        self.btn_recalculate = QPushButton("\u27f3  Recalculate")
        self.btn_recalculate.setMaximumWidth(160)
        self.btn_export_light = QPushButton("Export Plot (Light BG)")
        self.btn_export_light.setMaximumWidth(180)
        self.chk_top10 = QCheckBox("Restrict to Top 10 Satellites")
        self.lbl_status = QLabel("No data loaded")
        self.lbl_status.setStyleSheet(
            "color: #888888; font-style: italic; font-size: 11px;"
        )
        ctrl_row.addWidget(self.btn_recalculate)
        ctrl_row.addWidget(self.btn_export_light)
        ctrl_row.addWidget(self.chk_top10)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self.lbl_status, stretch=1)
        main_layout.addLayout(ctrl_row)

        # ---- Canvas (temporal mode, 2×2 grid) ----
        self.canvas = PValueCanvas()
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(NavigationToolbar(self.canvas, self))

    def show_message(self, msg: str) -> None:
        """Display a plain message (e.g. N_sat < 2 or no evolution file)."""
        self.canvas.clear_and_show_message(msg)
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(
            "color: #bb8800; font-style: italic; font-size: 11px;"
        )

    def update_plot(self, snap_nums, p_iso_dict, p_cosmo_dict, status: str = "") -> None:
        """
        Render the temporal p-value trajectories.

        Parameters
        ----------
        snap_nums : list[int]
        p_iso_dict : dict[str, list[float]]   metric → p_iso values per snap
        p_cosmo_dict : dict[str, list[float]] metric → p_cosmo values per snap
        status : str  shown in the status label
        """
        self.canvas.plot_temporal_significance(snap_nums, p_iso_dict, p_cosmo_dict)
        self.lbl_status.setText(status)
        self.lbl_status.setStyleSheet(
            "color: #aaaaaa; font-style: italic; font-size: 11px;"
        )


# --- Metric System Table Class ---
class MetricSystemTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.is_top10 = False

        self.tabs = QTabWidget()
        
        # --- Tab 1: Data Table ---
        self.tab_data = QWidget()
        self.tab_data_layout = QVBoxLayout(self.tab_data)
        
        self.table_widget = QTreeWidget()
        self.table_widget.setHeaderLabels(["Grouping / Halo ID", "Satellite Count"])
        self.table_widget.header().setSectionResizeMode(0, QHeaderView.Stretch)
        
        self.export_button = QPushButton("Export Current Table as CSV")
        self.export_button.clicked.connect(self.export_to_csv)
        self.export_button.setEnabled(False)
        
        self.tab_data_layout.addWidget(QLabel("Top/Bottom 1/3 Halos (Snap 99)"))
        self.tab_data_layout.addWidget(self.table_widget)
        self.tab_data_layout.addWidget(self.export_button)
        self.tabs.addTab(self.tab_data, "Data Table")
        
        # --- Tab 2: Satellite Histogram ---
        self.tab_hist = QWidget()
        self.tab_hist_layout = QVBoxLayout(self.tab_hist)
        
        hist_controls = QHBoxLayout()
        hist_controls.addWidget(QLabel("Select Metric:"))
        self.hist_metric_combo = QComboBox()
        self.hist_metric_combo.currentTextChanged.connect(self.plot_satellite_histogram)
        hist_controls.addWidget(self.hist_metric_combo)
        
        hist_controls.addWidget(QLabel("Binning Strategy:"))
        self.hist_binning_combo = QComboBox()
        self.hist_binning_combo.addItems(["Auto", "Integer", "Sturges", "Freedman-Diaconis"])
        self.hist_binning_combo.currentTextChanged.connect(self.plot_satellite_histogram)
        hist_controls.addWidget(self.hist_binning_combo)

        self.export_hist_button = QPushButton("Export Histogram (White BG)")
        self.export_hist_button.clicked.connect(self.export_histogram_white_bg)
        hist_controls.addWidget(self.export_hist_button)

        hist_controls.addStretch()
        
        self.tab_hist_layout.addLayout(hist_controls)
        
        self.hist_fig = Figure(figsize=(6, 4), dpi=100)
        self.hist_canvas = FigureCanvas(self.hist_fig)
        self.hist_ax = self.hist_fig.add_subplot(111)
        self._apply_dark_style_to_figure(self.hist_fig)
        self._apply_dark_style_to_axis(self.hist_ax, with_grid=True)
        
        self.hist_toolbar = NavigationToolbar(self.hist_canvas, self)
        
        self.tab_hist_layout.addWidget(self.hist_toolbar)
        self.tab_hist_layout.addWidget(self.hist_canvas)
        self.tabs.addTab(self.tab_hist, "Satellite Histogram")
        
        self.layout.addWidget(self.tabs)
        self.system_data = pd.DataFrame()

    def _apply_dark_style_to_figure(self, figure):
        figure.patch.set_facecolor("#1f1f1f")
        figure.set_facecolor("#1f1f1f")

    def _apply_light_style_to_figure(self, figure):
        figure.patch.set_facecolor("white")
        figure.set_facecolor("white")

    def _apply_dark_style_to_axis(self, axis, with_grid=True):
        axis.set_facecolor("#242424")
        axis.title.set_color("#e6e6e6")
        axis.xaxis.label.set_color("#e6e6e6")
        axis.yaxis.label.set_color("#e6e6e6")
        axis.tick_params(colors="#e6e6e6", which="both")
        for spine in axis.spines.values():
            spine.set_color("#5a5a5a")
        if with_grid:
            axis.grid(True, linestyle='--', alpha=0.5, color="#4a4a4a")

    def _apply_light_style_to_axis(self, axis, with_grid=True):
        axis.set_facecolor("white")
        axis.title.set_color("black")
        axis.xaxis.label.set_color("black")
        axis.yaxis.label.set_color("black")
        axis.tick_params(colors="black", which="both")
        for spine in axis.spines.values():
            spine.set_color("black")
        if with_grid:
            axis.grid(True, linestyle='--', alpha=0.5, color="gray")

    def set_top10_mode(self, is_top10):
        self.is_top10 = is_top10

    def update_table(self, data_df, snap99_sat_counts):
        self.snap99_sat_counts = snap99_sat_counts
        self.table_widget.clear()
        
        if data_df.empty:
            self.export_button.setEnabled(False)
            self.system_data = pd.DataFrame()
            self.hist_metric_combo.clear()
            return
            
        rows = []
        for _, row in data_df.iterrows():
            metric = row['Metric']
            top_ids = row['Most Lopsided (Top 1/3) Halo IDs (Snap 99)']
            bottom_ids = row['Least Lopsided (Bottom 1/3) Halo IDs (Snap 99)']
            
            top_list = [int(id_str) for id_str in top_ids.split(', ') if id_str.strip()] if isinstance(top_ids, str) else []
            bottom_list = [int(id_str) for id_str in bottom_ids.split(', ') if id_str.strip()] if isinstance(bottom_ids, str) else []
            
            for hid in top_list:
                sat_count = snap99_sat_counts.get(hid, np.nan)
                rows.append({"Metric": metric, "Category": "Top 1/3", "Halo ID": hid, "Satellite Count": sat_count})
                
            for hid in bottom_list:
                sat_count = snap99_sat_counts.get(hid, np.nan)
                rows.append({"Metric": metric, "Category": "Bottom 1/3", "Halo ID": hid, "Satellite Count": sat_count})
                
        self.system_data = pd.DataFrame(rows)
        
        # Build grouped GUI table
        self.table_widget.clear()
        
        if not self.system_data.empty:
            metrics = self.system_data["Metric"].unique()
            for metric in metrics:
                # Add Metric Section Header
                metric_item = QTreeWidgetItem(self.table_widget, [f"Metric: {metric}", ""])
                for col in range(2):
                    metric_item.setBackground(col, QColor("#3a4a62"))
                    metric_item.setForeground(col, QColor("#ffffff"))
                    font = metric_item.font(col)
                    font.setBold(True)
                    metric_item.setFont(col, font)
                
                df_metric = self.system_data[self.system_data["Metric"] == metric]
                for cat in ["Top 1/3", "Bottom 1/3"]:
                    df_cat = df_metric[df_metric["Category"] == cat]
                    if df_cat.empty:
                        continue
                        
                    # Add Category Sub-Section Header
                    cat_item = QTreeWidgetItem(metric_item, [f"--- {cat} ---", ""])
                    cat_item.setTextAlignment(0, Qt.AlignCenter)
                    for col in range(2):
                        cat_item.setBackground(col, QColor("#2d2d2d"))
                        cat_item.setForeground(col, QColor("#d3d3d3"))
                        font = cat_item.font(col)
                        font.setItalic(True)
                        cat_item.setFont(col, font)
                    
                    # Add Data Rows
                    for _, row in df_cat.iterrows():
                        sat_count_str = "N/A" if pd.isna(row["Satellite Count"]) else str(int(row["Satellite Count"]))
                        QTreeWidgetItem(cat_item, [str(row["Halo ID"]), sat_count_str])
                        
            self.table_widget.expandAll()
            
        self.export_button.setEnabled(True)
        
        # Update histogram metrics
        current_metric = self.hist_metric_combo.currentText()
        metrics = self.system_data["Metric"].unique().tolist()
        
        self.hist_metric_combo.blockSignals(True)
        self.hist_metric_combo.clear()
        self.hist_metric_combo.addItems(metrics)
        if current_metric in metrics:
            self.hist_metric_combo.setCurrentText(current_metric)
        elif metrics:
            self.hist_metric_combo.setCurrentIndex(0)
        self.hist_metric_combo.blockSignals(False)
        
        self.plot_satellite_histogram()

    def plot_satellite_histogram(self):
        self.hist_ax.clear()
        self._apply_dark_style_to_axis(self.hist_ax, with_grid=True)
        
        metric = self.hist_metric_combo.currentText()
        if not metric or self.system_data.empty:
            self.hist_ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=self.hist_ax.transAxes, color="#e6e6e6")
            self.hist_canvas.draw()
            return
            
        df_metric = self.system_data[self.system_data["Metric"] == metric]
        
        top_counts = df_metric[df_metric["Category"] == "Top 1/3"]["Satellite Count"].dropna()
        bottom_counts = df_metric[df_metric["Category"] == "Bottom 1/3"]["Satellite Count"].dropna()

        top_ids = set(df_metric[df_metric["Category"] == "Top 1/3"]["Halo ID"])
        bottom_ids = set(df_metric[df_metric["Category"] == "Bottom 1/3"]["Halo ID"])
        all_ids = set(getattr(self, 'snap99_sat_counts', {}).keys())
        remaining_ids = all_ids - top_ids - bottom_ids
        remaining_counts = pd.Series([
            self.snap99_sat_counts[hid]
            for hid in remaining_ids
            if not pd.isna(self.snap99_sat_counts.get(hid, np.nan))
        ])
        
        if top_counts.empty and bottom_counts.empty:
            self.hist_ax.text(0.5, 0.5, "No satellite counts available", ha='center', va='center', transform=self.hist_ax.transAxes, color="#e6e6e6")
            self.hist_canvas.draw()
            return
            
        binning_strategy = self.hist_binning_combo.currentText()
        
        # Determine bins based on all data
        all_counts = pd.concat([top_counts, bottom_counts, remaining_counts])
        bins = "auto"
        
        if binning_strategy == "Auto":
            bins = "auto"
        elif binning_strategy == "Sturges":
            bins = "sturges"
        elif binning_strategy == "Freedman-Diaconis":
            bins = "fd"
        elif binning_strategy == "Integer":
            if len(all_counts) > 0:
                min_val = int(all_counts.min())
                max_val = int(all_counts.max())
                bins = range(min_val, max_val + 2)
            else:
                bins = "auto"
                
        # Plot Remaining Systems (background)
        if not remaining_counts.empty:
            self.hist_ax.hist(remaining_counts, bins=bins, color="gray", alpha=0.4, label="Remaining Systems", edgecolor="black", zorder=1)

        # Plot Top 1/3
        if not top_counts.empty:
            self.hist_ax.hist(top_counts, bins=bins, color="red", alpha=0.6, label="Top 1/3 (Most Lopsided)", edgecolor="black", zorder=2)
            
        # Plot Bottom 1/3
        if not bottom_counts.empty:
            self.hist_ax.hist(bottom_counts, bins=bins, color="blue", alpha=0.6, label="Bottom 1/3 (Least Lopsided)", edgecolor="black", zorder=2)
            
        self.hist_ax.set_xlabel("Number of Satellites", color="#e6e6e6")
        self.hist_ax.set_ylabel("Frequency", color="#e6e6e6")
        self.hist_ax.set_title(f"Satellite Counts Distribution for {metric}", color="#e6e6e6")
        
        self.hist_ax.legend(fontsize="small")
        legend = self.hist_ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor("#2b2b2b")
            frame.set_edgecolor("#555555")
            frame.set_alpha(0.85)
            for text in legend.get_texts():
                text.set_color("#e6e6e6")
                
        self.hist_canvas.draw()

    def export_to_csv(self):
        if self.system_data.empty:
            QMessageBox.warning(self, "Export Failed", "No data to export.")
            return

        # Default filename uses today's date
        tag = "_top_10" if self.is_top10 else ""
        default_filename = f"Top_Bottom_Systems_Snap99_Metrics{tag}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"

        # Open file dialog to save CSV
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Metrics Table", default_filename, "CSV Files (*.csv)")

        if file_path:
            try:
                self.system_data.to_csv(file_path, index=False)
                QMessageBox.information(self, "Export Successful", f"Table exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save file: {e}")

    def export_histogram_white_bg(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Satellite Histogram",
            "Satellite_Histogram.png",
            "PNG Image (*.png);;All Files (*)"
        )
        if not filepath:
            return

        try:
            self._apply_light_style_to_figure(self.hist_fig)
            self._apply_light_style_to_axis(self.hist_ax, with_grid=True)
            legend = self.hist_ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor("white")
                frame.set_edgecolor("black")
                frame.set_alpha(1.0)
                for text in legend.get_texts():
                    text.set_color("black")
            for text in self.hist_ax.texts:
                text.set_color("black")

            self.hist_fig.savefig(filepath, facecolor='white', dpi=300, bbox_inches='tight')
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export histogram: {e}")
            return
        finally:
            self._apply_dark_style_to_figure(self.hist_fig)
            self._apply_dark_style_to_axis(self.hist_ax, with_grid=True)
            legend = self.hist_ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor("#2b2b2b")
                frame.set_edgecolor("#555555")
                frame.set_alpha(0.85)
                for text in legend.get_texts():
                    text.set_color("#e6e6e6")
            for text in self.hist_ax.texts:
                text.set_color("#e6e6e6")
            self.hist_canvas.draw()

        QMessageBox.information(self, "Export Successful", f"Histogram exported to:\n{filepath}")



# --- Potential Reasons View Class ---
class PotentialReasonsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_folder = None
        self.custom_corr_csv = None # Stores manually loaded CSV path
        self.df_merger = None
        self.image_paths = []
        self.current_img_idx = 0

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Image Gallery
        self.tab_gallery = QWidget()
        self.init_gallery_tab()
        self.tabs.addTab(self.tab_gallery, "Plot Gallery")

        # Tab 2: Interactive Plotter
        self.tab_plotter = QWidget()
        self.init_plotter_tab()
        self.tabs.addTab(self.tab_plotter, "Interactive Data Explorer")

        # Tab 3: Correlation Analysis (Hidden initially)
        self.tab_correlation = QWidget()
        self.init_correlation_tab()
        self.tabs.addTab(self.tab_correlation, "Correlation Analysis")
        self.tabs.setTabEnabled(2, False) # Disabled until analysis is run

    def _apply_dark_style_to_figure(self, figure):
        figure.patch.set_facecolor("#1f1f1f")
        figure.set_facecolor("#1f1f1f")

    def _apply_light_style_to_figure(self, figure):
        figure.patch.set_facecolor("white")
        figure.set_facecolor("white")

    def _apply_dark_style_to_axis(self, axis, with_grid=True):
        axis.set_facecolor("#242424")
        axis.title.set_color("#e6e6e6")
        axis.xaxis.label.set_color("#e6e6e6")
        axis.yaxis.label.set_color("#e6e6e6")
        axis.tick_params(colors="#e6e6e6", which="both")
        for spine in axis.spines.values():
            spine.set_color("#5a5a5a")
        if with_grid:
            axis.grid(True, linestyle='--', alpha=0.5, color="#4a4a4a")

    def _apply_light_style_to_axis(self, axis, with_grid=True):
        axis.set_facecolor("white")
        axis.title.set_color("black")
        axis.xaxis.label.set_color("black")
        axis.yaxis.label.set_color("black")
        axis.tick_params(colors="black", which="both")
        for spine in axis.spines.values():
            spine.set_color("black")
        if with_grid:
            axis.grid(True, linestyle='--', alpha=0.55, color="lightgray")

    def init_gallery_tab(self):
        """Sets up the gallery with Next/Prev buttons."""
        layout = QVBoxLayout(self.tab_gallery)

        # Container for the image and buttons
        controls_layout = QHBoxLayout()

        # Prev Button
        self.btn_prev = QPushButton("<")
        self.btn_prev.setFixedSize(40, 100)
        self.btn_prev.clicked.connect(self.show_prev_image)
        self.btn_prev.setEnabled(False)
        controls_layout.addWidget(self.btn_prev)

        # Image Label
        self.lbl_image = QLabel("Load a folder to see plots.")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("border: 1px solid #4a4a4a; background-color: #242424; color: #e6e6e6;")
        # Allow expansion
        self.lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        controls_layout.addWidget(self.lbl_image)

        # Next Button
        self.btn_next = QPushButton(">")
        self.btn_next.setFixedSize(40, 100)
        self.btn_next.clicked.connect(self.show_next_image)
        self.btn_next.setEnabled(False)
        controls_layout.addWidget(self.btn_next)

        layout.addLayout(controls_layout)

        # Filename Label at the bottom
        self.lbl_filename = QLabel("")
        self.lbl_filename.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_filename)

    def init_plotter_tab(self):
        """Sets up the list selector and matplotlib canvas."""
        layout = QVBoxLayout(self.tab_plotter)

        # Splitter allows resizing between list and plot
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # LEFT: List of Halo IDs
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(QLabel("Select Halo IDs (Hold Ctrl for multiple):"))

        self.halo_list = QListWidget()
        self.halo_list.setSelectionMode(QAbstractItemView.ExtendedSelection) # Allow multi-select
        self.halo_list.itemSelectionChanged.connect(self.update_plot)
        left_layout.addWidget(self.halo_list)

        splitter.addWidget(left_widget)

        # RIGHT: Matplotlib Canvas
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._apply_dark_style_to_figure(self.fig)
        self._apply_dark_style_to_axis(self.ax, with_grid=True)

        # Add Toolbar and Export Button
        toolbar_layout = QHBoxLayout()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.btn_export_history_plot = QPushButton("Export History Plot (White BG)")
        self.btn_export_history_plot.clicked.connect(self.export_history_plot_white_bg)
        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addWidget(self.btn_export_history_plot)
        right_layout.addLayout(toolbar_layout)
        right_layout.addWidget(self.canvas)

        splitter.addWidget(right_widget)

        # Set initial splitter sizes (30% list, 70% plot)
        splitter.setSizes([300, 700])

    def init_correlation_tab(self):
        """Sets up the correlation matrix view."""
        layout = QVBoxLayout(self.tab_correlation)

        # Data Source Control Panel
        source_panel = QHBoxLayout()
        self.lbl_corr_source = QLabel("Source: Auto-detecting...")
        self.lbl_corr_source.setStyleSheet("font-weight: bold; color: #d8d8d8;")

        self.btn_load_corr_csv = QPushButton("Select Custom Top/Bottom CSV")
        self.btn_load_corr_csv.clicked.connect(self.select_custom_corr_csv)

        source_panel.addWidget(self.lbl_corr_source)
        source_panel.addWidget(self.btn_load_corr_csv)
        source_panel.addStretch()

        layout.addLayout(source_panel)

        self.btn_export_corr_stats = QPushButton("Export Statistics CSV")
        self.btn_export_corr_stats.clicked.connect(self.export_correlation_stats_csv)
        self.btn_export_corr_stats.setEnabled(False)

        self.btn_export_heatmaps = QPushButton("Export Heatmaps (White BG)")
        self.btn_export_heatmaps.clicked.connect(self.export_heatmaps_white_bg)
        self.btn_export_heatmaps.setEnabled(False)

        contrib_controls = QHBoxLayout()
        contrib_label = QLabel("Data View:")
        self.corr_metric_selector = QComboBox()
        self.corr_metric_selector.currentTextChanged.connect(self._on_corr_metric_selector_changed)
        contrib_controls.addWidget(contrib_label)
        contrib_controls.addWidget(self.corr_metric_selector)
        contrib_controls.addStretch()
        contrib_controls.addWidget(self.btn_export_heatmaps)
        contrib_controls.addWidget(self.btn_export_corr_stats)
        layout.addLayout(contrib_controls)

        self.corr_class_table = QTableWidget(0, 6)
        self.corr_class_table.setHorizontalHeaderLabels(
            ["Merger_Class", "Top_Fraction", "Bottom_Fraction", "Difference", "Top_Residual", "Bottom_Residual"]
        )
        self.corr_class_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.corr_class_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.corr_class_table.setAlternatingRowColors(True)
        self.corr_class_table.verticalHeader().setVisible(False)
        self.corr_class_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.corr_class_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.corr_class_table.verticalHeader().setDefaultSectionSize(24)
        self.corr_class_table.verticalHeader().setMinimumSectionSize(22)
        self.corr_class_table.setMinimumHeight(110)
        self.corr_class_table.setMaximumHeight(220)

        class_header = self.corr_class_table.horizontalHeader()
        class_header.setSectionResizeMode(0, QHeaderView.Stretch)
        class_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        class_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        class_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        class_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        class_header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        class_header.setStretchLastSection(False)

        data_view_layout = QHBoxLayout()
        data_view_layout.addWidget(self.corr_class_table, 2)
        
        self.corr_class_details_by_metric = {}

        dist_layout = QVBoxLayout()
        dist_label = QLabel("Distribution Test Results (KS & Mann-Whitney):")
        dist_layout.addWidget(dist_label)
        
        self.dist_results_text = QTextEdit()
        self.dist_results_text.setReadOnly(True)
        self.dist_results_text.setMinimumHeight(110)
        self.dist_results_text.setMaximumHeight(220)
        dist_layout.addWidget(self.dist_results_text)
        
        data_view_layout.addLayout(dist_layout, 1)
        layout.addLayout(data_view_layout)

        self.corr_fig = Figure(figsize=(10, 6), dpi=100)
        self.corr_canvas = FigureCanvas(self.corr_fig)
        self.corr_toolbar = NavigationToolbar(self.corr_canvas, self)
        self._apply_dark_style_to_figure(self.corr_fig)

        layout.addWidget(self.corr_toolbar)
        layout.addWidget(self.corr_canvas)

    def _format_corr_stat(self, value, stat_name):
        if pd.isna(value):
            return "nan"

        val = float(value)
        if stat_name == "chi2":
            return f"{val:.4f}"
        if stat_name == "p_value":
            if val != 0.0 and abs(val) < 1e-4:
                return f"{val:.3e}"
            return f"{val:.4g}"
        if stat_name == "cramers_v":
            return f"{val:.4f}"
        return str(value)



    def _format_fraction_or_residual(self, value):
        if pd.isna(value):
            return "nan"
        return f"{float(value):.4f}"

    def _build_class_contribution_table(self, stat_result, class_order):
        """Build per-class Top/Bottom fractions and standardized residual columns."""
        class_fractions = stat_result.get("class_fractions", pd.DataFrame())
        standardized_residuals = stat_result.get("standardized_residuals", pd.DataFrame())
        residual_p_values = stat_result.get("residual_p_values", pd.DataFrame())

        available_classes = []
        if isinstance(class_fractions, pd.DataFrame) and not class_fractions.empty:
            available_classes.extend(list(class_fractions.index))
        if isinstance(standardized_residuals, pd.DataFrame) and not standardized_residuals.empty:
            available_classes.extend(list(standardized_residuals.columns))
        if isinstance(residual_p_values, pd.DataFrame) and not residual_p_values.empty:
            available_classes.extend(list(residual_p_values.columns))

        seen = set()
        unique_classes = []
        for class_name in available_classes:
            if class_name not in seen:
                seen.add(class_name)
                unique_classes.append(class_name)

        ordered_classes = [name for name in class_order if name in seen]
        ordered_classes.extend([name for name in unique_classes if name not in set(class_order)])

        rows = []
        for class_name in ordered_classes:
            top_fraction = float("nan")
            bottom_fraction = float("nan")
            fraction_diff = float("nan")
            if isinstance(class_fractions, pd.DataFrame) and class_name in class_fractions.index:
                top_fraction = class_fractions.at[class_name, "top_fraction"] if "top_fraction" in class_fractions.columns else float("nan")
                bottom_fraction = class_fractions.at[class_name, "bottom_fraction"] if "bottom_fraction" in class_fractions.columns else float("nan")
                fraction_diff = class_fractions.at[class_name, "fraction_diff"] if "fraction_diff" in class_fractions.columns else float("nan")

            top_residual = float("nan")
            bottom_residual = float("nan")
            if isinstance(standardized_residuals, pd.DataFrame):
                if "Top" in standardized_residuals.index and class_name in standardized_residuals.columns:
                    top_residual = standardized_residuals.at["Top", class_name]
                if "Bottom" in standardized_residuals.index and class_name in standardized_residuals.columns:
                    bottom_residual = standardized_residuals.at["Bottom", class_name]

            top_p_value = float("nan")
            bottom_p_value = float("nan")
            if isinstance(residual_p_values, pd.DataFrame):
                if "Top" in residual_p_values.index and class_name in residual_p_values.columns:
                    top_p_value = residual_p_values.at["Top", class_name]
                if "Bottom" in residual_p_values.index and class_name in residual_p_values.columns:
                    bottom_p_value = residual_p_values.at["Bottom", class_name]

            rows.append(
                {
                    "Merger_Class": class_name,
                    "Top_Fraction": top_fraction,
                    "Bottom_Fraction": bottom_fraction,
                    "Difference": fraction_diff,
                    "Top_Residual": top_residual,
                    "Top_p_value": top_p_value,
                    "Bottom_Residual": bottom_residual,
                    "Bottom_p_value": bottom_p_value,
                }
            )

        return pd.DataFrame(
            rows,
            columns=["Merger_Class", "Top_Fraction", "Bottom_Fraction", "Difference", "Top_Residual", "Top_p_value", "Bottom_Residual", "Bottom_p_value"],
        )

    def _update_correlation_class_table(self, metric_name):
        self.corr_class_table.clearContents()
        self.corr_class_table.setRowCount(0)
        self.corr_class_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.btn_export_corr_stats.setEnabled(False)

        is_overview = (metric_name == "All (Overview)")

        if is_overview:
            self.corr_class_table.setColumnCount(4)
            self.corr_class_table.setHorizontalHeaderLabels(["Metric", "chi2", "p_value", "Cramers_V"])
            df_to_show = getattr(self, "corr_stats_df", pd.DataFrame())
        else:
            self.corr_class_table.setColumnCount(8)
            self.corr_class_table.setHorizontalHeaderLabels(
                ["Merger_Class", "Top_Fraction", "Bottom_Fraction", "Difference", "Top_Residual", "Top_p_value", "Bottom_Residual", "Bottom_p_value"]
            )
            df_to_show = self.corr_class_details_by_metric.get(metric_name, pd.DataFrame())

        # Update Header Resize Modes
        class_header = self.corr_class_table.horizontalHeader()
        class_header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, self.corr_class_table.columnCount()):
            class_header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        class_header.setStretchLastSection(False)

        if df_to_show.empty:
            header_height = self.corr_class_table.horizontalHeader().height()
            frame_height = self.corr_class_table.frameWidth() * 2
            empty_height = header_height + self.corr_class_table.verticalHeader().defaultSectionSize() + frame_height + 4
            self.corr_class_table.setMinimumHeight(empty_height)
            self.corr_class_table.setMaximumHeight(empty_height)
            return

        self.btn_export_corr_stats.setEnabled(is_overview)
        self.corr_class_table.setRowCount(len(df_to_show))

        if is_overview:
            for row_idx, (_, row) in enumerate(df_to_show.iterrows()):
                self.corr_class_table.setItem(row_idx, 0, QTableWidgetItem(str(row.get("Metric", ""))))
                self.corr_class_table.setItem(row_idx, 1, QTableWidgetItem(self._format_corr_stat(row.get("chi2"), "chi2")))
                self.corr_class_table.setItem(row_idx, 2, QTableWidgetItem(self._format_corr_stat(row.get("p_value"), "p_value")))
                self.corr_class_table.setItem(row_idx, 3, QTableWidgetItem(self._format_corr_stat(row.get("cramers_v"), "cramers_v")))

                p_val_raw = row.get("p_value")
                is_significant = pd.notna(p_val_raw) and float(p_val_raw) < 0.05
                if is_significant:
                    bg_color = QColor("#364a36")
                    fg_color = QColor("#f0fff0")
                    for col_idx in range(4):
                        item = self.corr_class_table.item(row_idx, col_idx)
                        if item is not None:
                            item.setBackground(bg_color)
                            item.setForeground(fg_color)
        else:
            for row_idx, (_, row) in enumerate(df_to_show.iterrows()):
                self.corr_class_table.setItem(row_idx, 0, QTableWidgetItem(str(row.get("Merger_Class", ""))))
                self.corr_class_table.setItem(row_idx, 1, QTableWidgetItem(self._format_fraction_or_residual(row.get("Top_Fraction"))))
                self.corr_class_table.setItem(row_idx, 2, QTableWidgetItem(self._format_fraction_or_residual(row.get("Bottom_Fraction"))))
                self.corr_class_table.setItem(row_idx, 3, QTableWidgetItem(self._format_fraction_or_residual(row.get("Difference"))))
                self.corr_class_table.setItem(row_idx, 4, QTableWidgetItem(self._format_fraction_or_residual(row.get("Top_Residual"))))
                self.corr_class_table.setItem(row_idx, 5, QTableWidgetItem(self._format_corr_stat(row.get("Top_p_value"), "p_value")))
                self.corr_class_table.setItem(row_idx, 6, QTableWidgetItem(self._format_fraction_or_residual(row.get("Bottom_Residual"))))
                self.corr_class_table.setItem(row_idx, 7, QTableWidgetItem(self._format_corr_stat(row.get("Bottom_p_value"), "p_value")))

                bg_color = QColor("#364a36")
                fg_color = QColor("#f0fff0")
                
                top_p_val = row.get("Top_p_value")
                if pd.notna(top_p_val) and float(top_p_val) < 0.05:
                    item = self.corr_class_table.item(row_idx, 5)
                    if item is not None:
                        item.setBackground(bg_color)
                        item.setForeground(fg_color)
                
                bottom_p_val = row.get("Bottom_p_value")
                if pd.notna(bottom_p_val) and float(bottom_p_val) < 0.05:
                    item = self.corr_class_table.item(row_idx, 7)
                    if item is not None:
                        item.setBackground(bg_color)
                        item.setForeground(fg_color)

        self.corr_class_table.resizeRowsToContents()
        self.corr_class_table.resizeColumnsToContents()

        header_height = self.corr_class_table.horizontalHeader().height()
        frame_height = self.corr_class_table.frameWidth() * 2
        rows_height = sum(self.corr_class_table.rowHeight(i) for i in range(self.corr_class_table.rowCount()))
        target_height = header_height + rows_height + frame_height + 4

        max_no_scroll_rows = 8
        default_row_height = self.corr_class_table.verticalHeader().defaultSectionSize()
        visible_rows = min(max(self.corr_class_table.rowCount(), 1), max_no_scroll_rows)
        min_height = header_height + (default_row_height * visible_rows) + frame_height + 4
        if self.corr_class_table.rowCount() > max_no_scroll_rows:
            self.corr_class_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            max_height = header_height + (default_row_height * max_no_scroll_rows) + frame_height + 4
            target_height = min(target_height, max_height)

        target_height = max(min_height, min(target_height, 320))
        self.corr_class_table.setMinimumHeight(target_height)
        self.corr_class_table.setMaximumHeight(target_height)

    def _on_corr_metric_selector_changed(self, metric_name):
        self._update_correlation_class_table(metric_name)

    def _refresh_corr_metric_selector(self, metric_names):
        current_metric = self.corr_metric_selector.currentText()
        self.corr_metric_selector.blockSignals(True)
        self.corr_metric_selector.clear()

        combo_items = ["All (Overview)"] + list(metric_names) if metric_names else []
        self.corr_metric_selector.addItems(combo_items)

        if current_metric and current_metric in combo_items:
            self.corr_metric_selector.setCurrentText(current_metric)
        elif combo_items:
            self.corr_metric_selector.setCurrentText("All (Overview)")

        self.corr_metric_selector.blockSignals(False)
        self.corr_metric_selector.setEnabled(bool(combo_items))

        selected_metric = self.corr_metric_selector.currentText() if combo_items else ""
        self._update_correlation_class_table(selected_metric)

    def export_correlation_stats_csv(self):
        if not hasattr(self, "corr_stats_df") or self.corr_stats_df is None or self.corr_stats_df.empty:
            QMessageBox.warning(self, "Export Failed", "No correlation statistics available to export.")
            return

        export_df = self.corr_stats_df.loc[:, ["Metric", "chi2", "p_value", "cramers_v"]].copy()
        export_df = export_df.rename(columns={"cramers_v": "Cramers_V"})

        default_filename = f"Correlation_Statistics_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Correlation Statistics",
            default_filename,
            "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            export_df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export Successful", f"Statistics exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export statistics: {e}")

    def select_custom_corr_csv(self):
        """Allows manual selection of the Top/Bottom CSV file."""
        start_dir = self.data_folder if self.data_folder else os.getcwd()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Top/Bottom CSV", start_dir, "CSV Files (*.csv)")

        if file_path:
            self.custom_corr_csv = file_path
            self.generate_correlation_matrices()

    def load_data(self, folder_path):
        self.data_folder = folder_path
        self.load_gallery_images()
        self.load_csv_data()

    def load_gallery_images(self):
        """Loads all PNGs from the folder into the list."""
        self.image_paths = sorted(glob.glob(os.path.join(self.data_folder, "*.png")))
        self.current_img_idx = 0

        if not self.image_paths:
            self.lbl_image.setText("No PNG images found in folder.")
            self.lbl_filename.setText("")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            return

        self.update_gallery_image()

    def update_gallery_image(self):
        if not self.image_paths: return

        # Get current image
        img_path = self.image_paths[self.current_img_idx]
        pixmap = QPixmap(img_path)

        # Scale to fit label height, keeping aspect ratio
        h = self.lbl_image.height()
        if h > 10:
             pixmap = pixmap.scaledToHeight(h - 10, Qt.SmoothTransformation)

        self.lbl_image.setPixmap(pixmap)
        self.lbl_filename.setText(f"{os.path.basename(img_path)} ({self.current_img_idx + 1}/{len(self.image_paths)})")

        # Update Buttons
        self.btn_prev.setEnabled(self.current_img_idx > 0)
        self.btn_next.setEnabled(self.current_img_idx < len(self.image_paths) - 1)

    def show_prev_image(self):
        if self.current_img_idx > 0:
            self.current_img_idx -= 1
            self.update_gallery_image()

    def show_next_image(self):
        if self.current_img_idx < len(self.image_paths) - 1:
            self.current_img_idx += 1
            self.update_gallery_image()

    def resizeEvent(self, event):
        # Re-scale image on resize
        if self.image_paths:
             self.update_gallery_image()
        super().resizeEvent(event)

    def load_csv_data(self):
        """Loads processed_merger_data_with_ids.csv and populates the list."""
        # Search for filename pattern
        candidates = glob.glob(os.path.join(self.data_folder, "processed_merger_data_with_ids*.csv"))

        if not candidates:
            # Fallback to alternative naming
            candidates = glob.glob(os.path.join(self.data_folder, "processed_merger_data.csv"))

        if not candidates:
            QMessageBox.information(self, "Data Missing", "Could not find 'processed_merger_data_with_ids.csv' in this folder.\nPlease locate it manually.")
            file_path, _ = QFileDialog.getOpenFileName(self, "Select processed_merger_data CSV", self.data_folder, "CSV Files (*.csv)")
            if file_path:
                csv_path = file_path
            else:
                 QMessageBox.warning(self, "Action Cancelled", "Interactive features will be disabled.")
                 return
        else:
            # Use the first match (or latest modified if multiple)
            candidates.sort(key=os.path.getmtime)
            csv_path = candidates[-1]

        try:
            self.df_merger = pd.read_csv(csv_path)

            # Populate List Widget with unique Halo IDs
            unique_ids = sorted(self.df_merger['HaloID_Snap99'].unique())
            self.halo_list.clear()

            # Check if Classification exists
            has_class = 'Merger_Class' in self.df_merger.columns

            for hid in unique_ids:
                label = str(hid)
                if has_class:
                    # Get the classification for this halo (using first row of halo data)
                    cls = self.df_merger[self.df_merger['HaloID_Snap99'] == hid]['Merger_Class'].iloc[0]
                    if pd.notna(cls):
                        label = f"{hid} [{cls}]"
                self.halo_list.addItem(label)

            # If classification already exists, try to generate the correlation matrices immediately
            if has_class:
                self.generate_correlation_matrices()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")

    def update_plot(self):
        """Plots the selected Halos."""
        if self.df_merger is None:
            return

        selected_items = self.halo_list.selectedItems()
        # Parse ID from string "123 [Class]"
        selected_ids = []
        for item in selected_items:
            txt = item.text().split(' ')[0]
            if txt.isdigit():
                selected_ids.append(int(txt))

        self.ax.clear()
        self._apply_dark_style_to_axis(self.ax, with_grid=True)

        if not selected_ids:
            self.ax.text(0.5, 0.5, "Select a Halo ID to view history",
                         ha='center', va='center', transform=self.ax.transAxes, color="#e6e6e6")
            self.canvas.draw()
            return

        # Plot each selected halo
        for hid in selected_ids:
            halo_data = self.df_merger[self.df_merger['HaloID_Snap99'] == hid]

            # Ensure sorted by SnapNum
            halo_data = halo_data.sort_values('SnapNum')

            # Get class if available
            label = f"Halo {hid}"
            if 'Merger_Class' in halo_data.columns:
                c = halo_data['Merger_Class'].iloc[0]
                label += f" ({c})"

            self.ax.plot(halo_data['SnapNum'], halo_data['Mag_Gap'], marker='o', label=label)

        # Formatting
        self.ax.set_xlabel("Snapshot Number", color="#e6e6e6")
        self.ax.set_ylabel("Magnitude Gap (Sat - Central)", color="#e6e6e6")
        self.ax.set_title("Merger History (Magnitude Gap Evolution)", color="#e6e6e6")
        self.ax.axhspan(0, 1.5, color='gray', alpha=0.1, label='Major Merger Zone')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.legend()
        legend = self.ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor("#2b2b2b")
            frame.set_edgecolor("#555555")
            frame.set_alpha(0.85)
            for text in legend.get_texts():
                text.set_color("#e6e6e6")

        # Invert Y axis? Usually gap is positive, but if you want 0 at bottom:
        self.ax.set_ylim(-0.5, 5)

        self.canvas.draw()

    def generate_correlation_matrices(self):
        """
        Reads Top/Bottom metrics data, correlates with Merger Class,
        and plots 2 Heatmaps (Top 1/3 vs Bottom 1/3).
        """
        tb_file = None

        # 1. Use manual file if selected
        if self.custom_corr_csv:
            tb_file = self.custom_corr_csv

        # 2. Else auto-detect in PARENT folder
        elif self.data_folder:
            parent_dir = os.path.dirname(self.data_folder)
            pattern = os.path.join(parent_dir, "Top_Bottom_Systems_Snap99_*Metrics_*.csv")
            candidates = sorted(glob.glob(pattern))
            if candidates:
                tb_file = candidates[-1] # Latest file

        if not tb_file:
            self.lbl_corr_source.setText("Status: No Top/Bottom CSV found.")
            self.corr_stats_df = pd.DataFrame()
            self.corr_class_details_by_metric = {}
            self._refresh_corr_metric_selector([])
            if hasattr(self, 'dist_results_text'):
                self.dist_results_text.clear()
            if hasattr(self, 'btn_export_heatmaps'):
                self.btn_export_heatmaps.setEnabled(False)
            return

        self.lbl_corr_source.setText(f"Source: {os.path.basename(tb_file)}")

        try:
            df_tb = pd.read_csv(tb_file)
            corr_output = compute_correlation_matrices(
                df_top_bottom=df_tb,
                df_merger=self.df_merger,
                merger_classes=['Stable', 'Fly-by', 'Infall', 'Stripping'],
            )
            corr_results = corr_output["results"]
            merger_classes = corr_results["merger_classes"]
            metrics = corr_results["metrics"]
            matrix_top = corr_results["matrix_top"]
            matrix_bottom = corr_results["matrix_bottom"]
            norm_top = corr_results["norm_top"]
            norm_bottom = corr_results["norm_bottom"]

            contingency_inputs = build_contingency_inputs_from_top_bottom(
                df_top_bottom=df_tb,
                df_merger=self.df_merger,
            )
            stats_rows = []
            class_details_by_metric = {}
            for metric in metrics:
                contingency_input = contingency_inputs.get(metric, pd.DataFrame())
                if contingency_input.empty:
                    stats_rows.append(
                        {
                            "Metric": metric,
                            "chi2": float("nan"),
                            "p_value": float("nan"),
                            "cramers_v": float("nan"),
                        }
                    )
                    class_details_by_metric[metric] = pd.DataFrame(
                        columns=[
                            "Merger_Class",
                            "Top_Fraction",
                            "Bottom_Fraction",
                            "Difference",
                            "Top_Residual",
                            "Bottom_Residual",
                        ]
                    )
                    continue

                stat_result = compute_contingency_analysis(contingency_input)
                stats_rows.append(
                    {
                        "Metric": metric,
                        "chi2": stat_result["chi2"],
                        "p_value": stat_result["p_value"],
                        "cramers_v": stat_result["cramers_v"],
                    }
                )
                class_details_by_metric[metric] = self._build_class_contribution_table(stat_result, merger_classes)

            self.corr_stats_df = pd.DataFrame(stats_rows)
            self.corr_class_details_by_metric = class_details_by_metric
            self._refresh_corr_metric_selector(metrics)

            # --- New Feature Distribution Logic ---
            try:
                features_df = compute_merger_features(self.df_merger)
                
                target_metric = "Centroid_Offset"
                if target_metric not in contingency_inputs and metrics:
                    target_metric = metrics[0]

                dist_text = ""
                if target_metric in contingency_inputs:
                    flag_df = contingency_inputs[target_metric][["HaloID_Snap99", "lopsided_flag"]]
                    merged_df = features_df.merge(flag_df, on="HaloID_Snap99", how="inner")

                    dist_text += f"Distribution Tests for Top vs Bottom (Based on {target_metric}):\n"
                    dist_text += "=" * 60 + "\n\n"

                    for feature in ["time_since_last_major_merger", "mean_mag_gap"]:
                        if feature in merged_df.columns:
                            res = compare_distributions(merged_df, feature_column=feature)
                            dist_text += f"Feature: {feature}\n"
                            dist_text += f"  Kolmogorov-Smirnov: Stat = {res['ks_stat']:.4f}, p-value = {res['ks_p_value']:.4e}\n"
                            dist_text += f"  Mann-Whitney U:     Stat = {res['mw_stat']:.4f}, p-value = {res['mw_p_value']:.4e}\n"
                            dist_text += "-" * 60 + "\n"
                
                if not dist_text.strip():
                    dist_text = "No distribution data available."

                self.dist_results_text.setText(dist_text)
            except Exception as e:
                self.dist_results_text.setText(f"Error computing distribution tests: {e}")

            # 4. Plotting
            self.corr_fig.clear()
            self._apply_dark_style_to_figure(self.corr_fig)
            axes = self.corr_fig.subplots(1, 2)

            # Helper to draw heatmap
            def draw_heatmap(ax, count_matrix, norm_matrix, title):
                self._apply_dark_style_to_axis(ax, with_grid=False)
                # Plot using the CONTRAST normalized matrix (0-1 range)
                im = ax.imshow(norm_matrix, cmap='YlGnBu', vmin=0, vmax=1)

                # Ticks
                ax.set_xticks(np.arange(len(metrics)))
                ax.set_yticks(np.arange(len(merger_classes)))
                ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(merger_classes, fontsize=9)
                ax.tick_params(colors="#e6e6e6")

                # Text annotations (Counts)
                for i in range(len(merger_classes)):
                    for j in range(len(metrics)):
                        # Choose text color for contrast against background
                        val = norm_matrix[i, j]
                        text_color = "white" if val > 0.5 else "black"

                        text = ax.text(j, i, count_matrix[i, j],
                                       ha="center", va="center", color=text_color, fontweight='bold')

                ax.set_title(title, fontsize=10, fontweight='bold', color="#e6e6e6")
                return im

            draw_heatmap(axes[0], matrix_top, norm_top, "Composition of Most Lopsided (Top 1/3)")
            draw_heatmap(axes[1], matrix_bottom, norm_bottom, "Composition of Least Lopsided (Bottom 1/3)")

            self.corr_fig.tight_layout()
            self.corr_canvas.draw()

            # Enable the tab
            self.tabs.setTabEnabled(2, True)
            if hasattr(self, 'btn_export_heatmaps'):
                self.btn_export_heatmaps.setEnabled(True)

        except Exception as e:
            print(f"Error generating correlation matrices: {e}")
            self.corr_stats_df = pd.DataFrame()
            self.corr_class_details_by_metric = {}
            self._refresh_corr_metric_selector([])
            if hasattr(self, 'dist_results_text'):
                self.dist_results_text.clear()
            if hasattr(self, 'btn_export_heatmaps'):
                self.btn_export_heatmaps.setEnabled(False)

    def export_history_plot_white_bg(self):
        selected_items = self.halo_list.selectedItems()
        if not selected_items or self.df_merger is None:
            QMessageBox.warning(self, "Export Failed", "No halo history selected or available to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export History Plot",
            "Merger_Magnitude_Gap_White.png",
            "PNG Image (*.png);;All Files (*)"
        )
        if not filepath:
            return

        try:
            self._apply_light_style_to_figure(self.fig)
            self._apply_light_style_to_axis(self.ax, with_grid=True)
            self.ax.xaxis.label.set_color("black")
            self.ax.yaxis.label.set_color("black")
            self.ax.title.set_color("black")
            legend = self.ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor("white")
                frame.set_edgecolor("black")
                frame.set_alpha(1.0)
                for text in legend.get_texts():
                    text.set_color("black")
            self.fig.savefig(filepath, facecolor='white', dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", f"Saved history plot to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export history plot: {e}")
        finally:
            self._apply_dark_style_to_figure(self.fig)
            self._apply_dark_style_to_axis(self.ax, with_grid=True)
            self.ax.xaxis.label.set_color("#e6e6e6")
            self.ax.yaxis.label.set_color("#e6e6e6")
            self.ax.title.set_color("#e6e6e6")
            legend = self.ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor("#2b2b2b")
                frame.set_edgecolor("#555555")
                frame.set_alpha(0.85)
                for text in legend.get_texts():
                    text.set_color("#e6e6e6")
            self.canvas.draw()

    def export_heatmaps_white_bg(self):
        if not self.corr_fig.axes:
            QMessageBox.warning(self, "Export Failed", "No correlation heatmaps available to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Correlation Heatmaps",
            "Merger_Correlation_Heatmaps_White.png",
            "PNG Image (*.png);;All Files (*)"
        )
        if not filepath:
            return

        try:
            self._apply_light_style_to_figure(self.corr_fig)
            for ax in self.corr_fig.axes:
                self._apply_light_style_to_axis(ax, with_grid=False)
                ax.title.set_color("black")
                ax.tick_params(colors="black")
                for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                    lbl.set_color("black")
            self.corr_fig.savefig(filepath, facecolor='white', dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", f"Saved correlation heatmaps to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export correlation heatmaps: {e}")
        finally:
            self._apply_dark_style_to_figure(self.corr_fig)
            for ax in self.corr_fig.axes:
                self._apply_dark_style_to_axis(ax, with_grid=False)
                ax.title.set_color("#e6e6e6")
                ax.tick_params(colors="#e6e6e6")
                for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                    lbl.set_color("#e6e6e6")
            self.corr_canvas.draw()


# --- Main Application Window ---
class HaloReaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Halo Track Viewer")
        self.setWindowIcon(QIcon("COSMICOR-SlackLogo.ico"))
        self._apply_dark_theme()

        # --- Data holders ---
        self.base_export_folder = None
        self.lineage_df = pd.DataFrame()
        self.current_lineage = {}
        self.lineage_metrics = {}
        self.system_table_data = {} # Stores dynamic Top/Bottom Halo IDs for the table
        self.top10_systems_map = {} # Maps snap -> set of halo IDs
        # Store Lineage DF for plotting reference
        self.lineage_for_plotting = pd.DataFrame()
        # --- p-value engine (initialised after base_export_folder is set) ---
        self.p_value_engine: IsotropicNullEngine | None = None
        self._cosmo_ensemble_cache: dict = {}  # (folder, snap) → pd.DataFrame memo
        self._current_halo_dir: str | None = None  # for recalculate/top10 handlers
        self._p_value_dirty = True
        self._sat_centric_dirty = True
        self._cached_p_val_files = []

        # --- Main Layout ---
        self.layout = QHBoxLayout()
        self.left_side_layout = QVBoxLayout()

        # --- Mode Selection ---
        self.mode_group = QGroupBox("Select View Mode")
        self.mode_layout = QVBoxLayout()
        self.mode_single_halo = QRadioButton("Single Halo View")
        self.mode_tracked_history = QRadioButton("Tracked Halo History")
        self.mode_metrics_overview = QRadioButton("Metrics Overview (Lineage Analysis)") # RENAMED
        self.mode_potential_reasons = QRadioButton("Potential Reasons") # NEW MODE
        self.mode_single_halo.setChecked(True)
        self.mode_layout.addWidget(self.mode_single_halo)
        self.mode_layout.addWidget(self.mode_tracked_history)
        self.mode_layout.addWidget(self.mode_metrics_overview)
        self.mode_layout.addWidget(self.mode_potential_reasons)

        # Checkbox for Top 10 Only (Global for modes that use it)
        self.toggle_top10 = QCheckBox("Show Top 10 Only")
        # Connect signal later or in update logic
        self.mode_layout.addWidget(self.toggle_top10)

        self.mode_group.setLayout(self.mode_layout)
        self.left_side_layout.addWidget(self.mode_group)
        self.mode_single_halo.toggled.connect(self.update_ui_for_mode)
        self.mode_tracked_history.toggled.connect(self.update_ui_for_mode)
        self.mode_metrics_overview.toggled.connect(self.update_ui_for_mode)
        self.mode_potential_reasons.toggled.connect(self.update_ui_for_mode)

        # Connect Top 10 toggle to update views
        self.toggle_top10.stateChanged.connect(self.on_top10_toggled)

        # --- Controls for "Single Halo View" ---
        self.single_halo_group = QGroupBox("Data Selection (SnapXX/Halo ID)")
        self.single_halo_layout = QVBoxLayout()
        self.file_list = QListWidget()
        self.select_folder_button = QPushButton("Select Halo Folder")
        self.select_folder_button.clicked.connect(lambda: self.select_folder(prompt_text="Select Specific Halo Folder (e.g., .../Snap99/Halo 488)"))
        self.focus_reset_button = QPushButton("Back to Default View")
        self.focus_reset_button.clicked.connect(self.back_to_default)
        self.focus_reset_button.hide()
        self.single_halo_layout.addWidget(self.file_list)
        self.single_halo_layout.addWidget(self.select_folder_button)
        self.single_halo_layout.addWidget(self.focus_reset_button)

        self.single_halo_group.setLayout(self.single_halo_layout)
        self.left_side_layout.addWidget(self.single_halo_group)

        # --- Controls for "Tracked Halo History" ---
        self.history_group = QGroupBox("Lineage Controls")
        self.history_layout = QVBoxLayout()
        self.history_folder_button = QPushButton("Select Base Export Folder")
        self.history_folder_button.clicked.connect(lambda: self.select_folder(prompt_text="Select Base Export Folder (containing Snap folders)"))
        self.history_layout.addWidget(self.history_folder_button)
        self.history_layout.addWidget(QLabel("1. Select Lineage (by Halo ID at latest snap):"))
        self.lineage_combo = QComboBox()
        self.lineage_combo.currentIndexChanged.connect(self.on_lineage_selected)
        self.history_layout.addWidget(self.lineage_combo)
        self.history_layout.addWidget(QLabel("2. Select Snapshot to View in 3D:"))
        self.snapshot_slider = QSlider(Qt.Horizontal)
        self.snapshot_slider.setTickPosition(QSlider.TicksBelow)
        self.snapshot_slider.setTickInterval(1)
        self.snapshot_slider.setSingleStep(1)
        self.snapshot_slider.valueChanged.connect(self.update_3d_view_for_lineage_snap)
        self.snapshot_label = QLabel("N/A")
        self.history_layout.addWidget(self.snapshot_slider)
        self.history_layout.addWidget(self.snapshot_label, alignment=Qt.AlignCenter)
        self.history_group.setLayout(self.history_layout)
        self.left_side_layout.addWidget(self.history_group)

        # --- Controls for "Metrics Overview" ---
        self.overview_group = QGroupBox("Overview Controls")
        self.overview_layout = QVBoxLayout()
        self.overview_folder_button = QPushButton("Select Base Export Folder and Plot")
        self.overview_folder_button.clicked.connect(lambda: self.select_folder(prompt_text="Select Base Export Folder (containing Snap folders)"))
        self.overview_layout.addWidget(self.overview_folder_button)

        self.export_all_lines_button = QPushButton("Export All Individual Tracks Plot")
        self.export_all_lines_button.clicked.connect(self.export_all_lines_plot)
        self.export_all_lines_button.setEnabled(False) # Disabled until data is loaded
        self.overview_layout.addWidget(self.export_all_lines_button)
        self.overview_group.setLayout(self.overview_layout)
        self.left_side_layout.addWidget(self.overview_group)

        # --- Controls for "Potential Reasons" ---
        self.reasons_group = QGroupBox("Potential Reasons")
        self.reasons_layout = QVBoxLayout()
        self.reasons_layout.addWidget(QLabel("Select Reason Category:"))
        self.reason_selector = QComboBox()
        self.reason_selector.addItem("Merger History")
        self.reasons_layout.addWidget(self.reason_selector)
        self.reasons_folder_button = QPushButton("Select Data Folder")
        self.reasons_folder_button.clicked.connect(self.select_reasons_folder)
        self.reasons_layout.addWidget(self.reasons_folder_button)
        self.lbl_reasons_status = QLabel("No folder selected")
        self.lbl_reasons_status.setStyleSheet("color: gray; font-style: italic;")
        self.reasons_layout.addWidget(self.lbl_reasons_status)
        self.reasons_group.setLayout(self.reasons_layout)
        self.left_side_layout.addWidget(self.reasons_group)

        # --- Common Controls & Displays ---
        self.metrics_label = QLabel("Metrics for current view:")
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.subhalo_info_label = QLabel("Subhalo Info:")
        self.subhalo_info_label.hide()
        self.subhalo_info_text = QTextEdit()
        self.subhalo_info_text.setReadOnly(True)
        self.subhalo_info_text.hide()
        self.left_side_layout.addWidget(self.metrics_label)
        self.left_side_layout.addWidget(self.metrics_text)
        self.left_side_layout.addWidget(self.subhalo_info_label)
        self.left_side_layout.addWidget(self.subhalo_info_text)

        self.toggle_dark = QCheckBox("Show Dark Subhalos")
        self.toggle_dark.setChecked(True)
        self.toggle_tracks = QCheckBox("Show Subhalo Tracks")
        self.toggle_tracks.setChecked(True)
        self.toggle_box = QCheckBox("Show Simulation Box")
        self.toggle_box.setChecked(False)
        self.toggle_dark.stateChanged.connect(self.update_current_view)
        self.toggle_tracks.stateChanged.connect(self.update_current_view)
        self.toggle_box.stateChanged.connect(self.update_current_view)
        self.left_side_layout.addWidget(self.toggle_dark)
        self.left_side_layout.addWidget(self.toggle_tracks)
        self.left_side_layout.addWidget(self.toggle_box)

        # --- Clear Selection button ---
        self.clear_button = QPushButton("Clear Selection")
        self.clear_button.clicked.connect(self.clear_all)
        self.left_side_layout.addWidget(self.clear_button)

        # --- Right Side Layout (Plots) ---
        self.plot_tabs = QTabWidget()
        plot_3d_tab, self.canvas_3d, self.toolbar = self.create_3d_plot_tab()
        self.plot_tabs.addTab(plot_3d_tab, "3D View")

        (
            metric_history_tab,
            self.canvas_metrics,
            self.toolbar_metrics,
            self.btn_export_lineage_core,
            self.btn_export_lineage_all,
        ) = self.create_metric_plot_tab()
        self.btn_export_lineage_core.clicked.connect(self.export_lineage_core)
        self.btn_export_lineage_all.clicked.connect(self.export_lineage_all)
        self.plot_tabs.addTab(metric_history_tab, "Lineage Metric Evolution")

        # --- Overview Canvases (Now separate tabs instead of nested tabs) ---
        self.canvas_overview_satellite = MetricOverviewCanvas()
        satellite_centric_tab = QWidget()
        sat_layout = QVBoxLayout(satellite_centric_tab)
        sat_layout.addWidget(self.canvas_overview_satellite)
        sat_toolbar_layout = QHBoxLayout()
        sat_toolbar = NavigationToolbar(self.canvas_overview_satellite, self)
        self.btn_export_overview_satellite = QPushButton("Export Overview (White BG)")
        self.btn_export_overview_satellite.clicked.connect(self.export_overview_satellite_white_bg)
        sat_toolbar_layout.addWidget(sat_toolbar)
        sat_toolbar_layout.addWidget(self.btn_export_overview_satellite)
        sat_layout.addLayout(sat_toolbar_layout)
        self.plot_tabs.addTab(satellite_centric_tab, "Overview: Satellite-Centric (Fixed)")

        self.canvas_overview_host = MetricOverviewCanvas()
        host_centric_tab = QWidget()
        host_layout = QVBoxLayout(host_centric_tab)
        host_layout.addWidget(self.canvas_overview_host)
        host_toolbar_layout = QHBoxLayout()
        host_toolbar = NavigationToolbar(self.canvas_overview_host, self)
        self.btn_export_overview_host = QPushButton("Export Overview (White BG)")
        self.btn_export_overview_host.clicked.connect(self.export_overview_host_white_bg)
        host_toolbar_layout.addWidget(host_toolbar)
        host_toolbar_layout.addWidget(self.btn_export_overview_host)
        host_layout.addLayout(host_toolbar_layout)
        self.plot_tabs.addTab(host_centric_tab, "Overview: Host-Centric (Evolving)")

        # --- Top/Bottom Systems Table Tab ---
        self.canvas_systems_table = MetricSystemTable()
        systems_table_tab = QWidget()
        table_layout = QVBoxLayout(systems_table_tab)
        table_layout.addWidget(self.canvas_systems_table)
        self.plot_tabs.addTab(systems_table_tab, "Top/Bottom Systems (Snap 99)")

        # --- Snapshot 99 Corner Plot Tab ---
        self.corner_plot_tab = QWidget()
        corner_tab_layout = QHBoxLayout(self.corner_plot_tab)
        corner_tab_layout.setContentsMargins(4, 4, 4, 4)
        corner_tab_layout.setSpacing(6)

        # Left control panel
        corner_ctrl_panel = QWidget()
        corner_ctrl_panel.setMaximumWidth(220)
        corner_ctrl_layout = QVBoxLayout(corner_ctrl_panel)
        corner_ctrl_layout.setContentsMargins(4, 4, 4, 4)
        corner_ctrl_layout.setSpacing(6)

        corner_ctrl_layout.addWidget(QLabel("1. Categorize (Red/Blue) by:"))
        self.corner_category_combo = QComboBox()
        corner_ctrl_layout.addWidget(self.corner_category_combo)

        corner_ctrl_layout.addWidget(QLabel("2. Metric Order (Drag to reorder):"))
        self.corner_metric_list = QListWidget()
        self.corner_metric_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.corner_metric_list.setMaximumHeight(150)
        corner_ctrl_layout.addWidget(self.corner_metric_list)

        self.corner_pct_checkbox = QCheckBox("Plot Empirical P-Values (Percentiles)")
        self.corner_pct_checkbox.stateChanged.connect(self.update_corner_plot)
        corner_ctrl_layout.addWidget(self.corner_pct_checkbox)

        corner_apply_btn = QPushButton("Apply && Update Plot")
        corner_apply_btn.clicked.connect(self.update_corner_plot)
        corner_ctrl_layout.addWidget(corner_apply_btn)

        self.btn_export_corner = QPushButton("Export Corner Plot (White BG)")
        self.btn_export_corner.clicked.connect(self.export_corner_plot_white_bg)
        corner_ctrl_layout.addWidget(self.btn_export_corner)
        corner_ctrl_layout.addStretch()

        # Right canvas area
        corner_canvas_widget = QWidget()
        self.corner_canvas_layout = QVBoxLayout(corner_canvas_widget)
        self.corner_canvas_layout.setContentsMargins(0, 0, 0, 0)

        corner_tab_layout.addWidget(corner_ctrl_panel, stretch=1)
        corner_tab_layout.addWidget(corner_canvas_widget, stretch=4)
        self.plot_tabs.addTab(self.corner_plot_tab, "Overview: Snap 99 Corner Plot")

        # --- Potential Reasons Tab ---
        self.view_potential_reasons = PotentialReasonsView()
        self.plot_tabs.addTab(self.view_potential_reasons, "Potential Reasons Analysis")

        # --- Statistical Significance Tab (tab 7) ---
        self.p_value_tab = PValueTab()
        self.plot_tabs.addTab(self.p_value_tab, "Statistical Significance (p-values)")
        self.p_value_tab.btn_recalculate.clicked.connect(self._on_pvalue_recalculate)
        self.p_value_tab.btn_export_light.clicked.connect(self.export_p_value_plot_light)
        self.p_value_tab.chk_top10.stateChanged.connect(self._on_pvalue_top10_toggled)
        # ----------------------------------------

        # --- Satellite-Centric Significance Evolution Tab (tab 8) ---
        self.sat_centric_tab = SatCentricTab()
        self.plot_tabs.addTab(self.sat_centric_tab, "Sat-Centric Significance Evolution")
        self.sat_centric_tab.btn_recalculate.clicked.connect(self._on_sat_centric_recalculate)
        self.sat_centric_tab.btn_export_light.clicked.connect(self.export_sat_centric_plot_light)
        self.sat_centric_tab.chk_top10.stateChanged.connect(self._on_sat_centric_top10_toggled)
        # ----------------------------------------

        # Connect tab switching for lazy evaluation of p-value tabs
        self.plot_tabs.currentChanged.connect(self._on_plot_tab_changed)

        self.layout.addLayout(self.left_side_layout, stretch=2)
        self.layout.addWidget(self.plot_tabs, stretch=7)
        self.setLayout(self.layout)
        self._init_p_value_engine()   # initialise engine with fallback cwd cache
        self.update_ui_for_mode()


    def _apply_dark_theme(self):
        """Apply a dark default UI theme when the app opens."""
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1e1e1e;
                color: #e6e6e6;
            }
            QWidget:disabled {
                color: #9a9a9a;
            }
            QGroupBox {
                border: 1px solid #3a3a3a;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
            }
            QPushButton {
                background-color: #2f2f2f;
                border: 1px solid #4a4a4a;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #242424;
                border: 1px solid #3d3d3d;
                color: #8f8f8f;
            }
            QLineEdit, QTextEdit, QListWidget, QComboBox {
                background-color: #252525;
                border: 1px solid #444444;
                color: #e6e6e6;
            }
            QLineEdit:disabled, QTextEdit:disabled, QListWidget:disabled, QComboBox:disabled {
                background-color: #212121;
                border: 1px solid #3a3a3a;
                color: #8f8f8f;
            }
            QTableWidget, QTableView {
                background-color: #232323;
                alternate-background-color: #2a2a2a;
                color: #e6e6e6;
                border: 1px solid #3f3f3f;
                gridline-color: #474747;
                selection-background-color: #3a4a62;
                selection-color: #f5f8ff;
            }
            QTableWidget::item, QTableView::item {
                padding: 4px;
                color: #e6e6e6;
            }
            QTableWidget::item:selected, QTableView::item:selected {
                background-color: #3a4a62;
                color: #f5f8ff;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #f0f0f0;
                border: 1px solid #474747;
                padding: 5px 6px;
                font-weight: 600;
            }
            QTableCornerButton::section {
                background-color: #2d2d2d;
                border: 1px solid #474747;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
            }
            QTabBar::tab {
                border: 1px solid #4a4a4a;
                border-bottom: none;
                padding: 6px 10px;
                margin-right: 2px;
                min-width: 90px;
            }
            QTabBar::tab:selected {
                background-color: #343434;
                color: #f2f2f2;
            }
            QTabBar::tab:!selected {
                background-color: #252525;
                color: #c5c5c5;
            }
            QTabBar::tab:hover {
                background-color: #3a3a3a;
                color: #ffffff;
            }
            QTabBar::tab:disabled {
                background-color: #222222;
                color: #8f8f8f;
            }
            QRadioButton, QCheckBox {
                spacing: 6px;
                color: #e6e6e6;
            }
            QRadioButton:disabled, QCheckBox:disabled {
                color: #8f8f8f;
            }
            QRadioButton::indicator, QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #7a7a7a;
                background-color: #232323;
            }
            QRadioButton::indicator {
                border-radius: 7px;
            }
            QCheckBox::indicator {
                border-radius: 2px;
            }
            QRadioButton::indicator:checked, QCheckBox::indicator:checked {
                background-color: #5f8fd0;
                border: 1px solid #8eb2e3;
            }
            QRadioButton::indicator:unchecked, QCheckBox::indicator:unchecked {
                background-color: #232323;
                border: 1px solid #7a7a7a;
            }
            QRadioButton::indicator:disabled, QCheckBox::indicator:disabled {
                background-color: #1f1f1f;
                border: 1px solid #5a5a5a;
            }
            QComboBox {
                padding-right: 20px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid #4a4a4a;
                background-color: #2d2d2d;
            }
            QComboBox::down-arrow {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #232323;
                color: #e6e6e6;
                border: 1px solid #4a4a4a;
                selection-background-color: #3a4a62;
                selection-color: #f5f8ff;
            }
            QToolBar {
                background-color: #232323;
                border: 1px solid #3a3a3a;
                spacing: 3px;
            }
            QToolBar::separator {
                background-color: #4a4a4a;
                width: 1px;
                margin: 3px 4px;
            }
            QToolButton {
                background-color: #2f2f2f;
                border: 1px solid #4a4a4a;
                color: #e6e6e6;
                padding: 3px 6px;
            }
            QToolButton:hover {
                background-color: #3a3a3a;
            }
            QToolButton:pressed {
                background-color: #1f1f1f;
            }
            QToolButton:disabled {
                background-color: #232323;
                border: 1px solid #3a3a3a;
                color: #8f8f8f;
            }
            """
        )


    # ==========================================================================
    # Busy State Management & Event Pump
    # ==========================================================================

    @contextmanager
    def busy_state(self, status_msg: str = "Processing..."):
        """
        Context manager for responsive busy state management:
        - Locks GUI interactive control groups to prevent clicks and race conditions.
        - Changes mouse cursor to Qt.WaitCursor.
        - Updates window title with status message.
        - Safely restores controls, original title, and cursor even on exceptions.
        - Supports re-entrancy / nesting.
        """
        depth = getattr(self, "_busy_depth", 0)
        self._busy_depth = depth + 1

        if depth == 0:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._busy_orig_title = self.windowTitle()

        self.setWindowTitle(f"Halo Track Viewer — [{status_msg}]")

        # Disable interactive control groups to block clicks
        self.mode_group.setEnabled(False)
        self.plot_tabs.setEnabled(False)
        if hasattr(self, "single_halo_group"):
            self.single_halo_group.setEnabled(False)
        if hasattr(self, "history_group"):
            self.history_group.setEnabled(False)
        if hasattr(self, "overview_group"):
            self.overview_group.setEnabled(False)
        if hasattr(self, "reasons_group"):
            self.reasons_group.setEnabled(False)

        QApplication.processEvents()
        self._last_event_pump = time.perf_counter()

        try:
            yield self
        finally:
            self._busy_depth = max(0, getattr(self, "_busy_depth", 1) - 1)
            if self._busy_depth == 0:
                self.mode_group.setEnabled(True)
                self.plot_tabs.setEnabled(True)
                if hasattr(self, "single_halo_group"):
                    self.single_halo_group.setEnabled(True)
                if hasattr(self, "history_group"):
                    self.history_group.setEnabled(True)
                if hasattr(self, "overview_group"):
                    self.overview_group.setEnabled(True)
                if hasattr(self, "reasons_group"):
                    self.reasons_group.setEnabled(True)

                self.setWindowTitle(getattr(self, "_busy_orig_title", "Halo Track Viewer"))
                QApplication.restoreOverrideCursor()
                QApplication.processEvents()

    def pump_events(self, min_interval_sec: float = 0.05):
        """
        Throttled Qt/Windows event pump: calling processEvents() every iteration
        can slow down tight loops, so limit it to once every ~50ms.
        """
        now = time.perf_counter()
        if not hasattr(self, "_last_event_pump") or (now - self._last_event_pump) >= min_interval_sec:
            QApplication.processEvents()
            self._last_event_pump = now

    # ==========================================================================
    # p-Value Engine — cache management and data extraction helpers
    # ==========================================================================

    def _get_cache_dir(self) -> str:
        """
        Return the path to the ``.p_val_cache`` directory.

        Uses ``base_export_folder`` if set; falls back to ``os.getcwd()``.
        The directory is created if it does not yet exist.
        """
        base = self.base_export_folder if self.base_export_folder else os.getcwd()
        cache_dir = os.path.join(base, ".p_val_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _init_p_value_engine(self) -> None:
        """
        Instantiate (or re-instantiate) the ``IsotropicNullEngine`` bound to
        the correct ``.p_val_cache`` directory.

        Called once at startup and again whenever ``base_export_folder`` changes.
        Preserves existing in-memory distributions from the old engine so that
        a folder switch doesn't throw away already-computed results.
        """
        cache_dir = self._get_cache_dir()
        self.p_value_engine = IsotropicNullEngine(cache_dir)
        # Invalidate the in-process cosmological ensemble memo whenever
        # the folder changes so stale data cannot leak across sessions.
        invalidate_cosmo_ensemble_cache()

    def _get_metrics_from_files(self, files: list, top10: bool = False) -> dict | None:
        """
        Scan *files* for a metrics CSV and extract values needed for p-value
        computation.

        When *top10* is True, prefers ``*_metrics_top_10.csv``; falls back to
        the full ``*_metrics.csv`` for ``Group_R_Crit200`` (the top-10 files
        do not carry that column).

        Returns
        -------
        dict or None
            Keys: ``"MRL"``, ``"Pairwise"``, ``"Hemi_Contrast"``,
            ``"Centroid_Offset"``, ``"N_sat"``, ``"Group_R_Crit200"``,
            ``"snap_num"``, ``"metrics_file_path"``.
            Returns ``None`` if no suitable metrics file is found.
        """
        if top10:
            # Prefer top-10 file for metric values
            metrics_file = next(
                (f for f in files
                 if "metrics_top_10" in os.path.basename(f).lower()
                 and f.endswith(".csv")),
                None,
            )
        else:
            metrics_file = None

        # Always also locate the full metrics file for Group_R_Crit200
        full_metrics_file = next(
            (f for f in files
             if "metrics" in os.path.basename(f).lower()
             and "top_10" not in os.path.basename(f).lower()
             and f.endswith(".csv")),
            None,
        )

        # Fall back to full file if top-10 not found
        if metrics_file is None:
            metrics_file = full_metrics_file
        if metrics_file is None or not os.path.exists(metrics_file):
            return None

        try:
            row = pd.read_csv(metrics_file).iloc[0].to_dict()
        except Exception:
            return None

        # Group_R_Crit200 only exists in the full metrics file
        r200 = float(row.get("Group_R_Crit200", 0.0))
        if r200 == 0.0 and full_metrics_file and full_metrics_file != metrics_file:
            try:
                full_row = pd.read_csv(full_metrics_file).iloc[0].to_dict()
                r200 = float(full_row.get("Group_R_Crit200", 0.0))
            except Exception:
                pass

        # N_sat column name differs between full and top-10 files
        n_sat = int(row.get("visible_subhalos", row.get("subhalo_count", 0)))

        # Centroid key: some files use 'centroid', others 'Centroid_Distance'
        centroid_val = row.get("centroid", row.get("Centroid_Distance", float("nan")))

        # Parse snapshot number from file name
        snap_num = None
        match = re.search(r"_snap_(\d+)", os.path.basename(metrics_file))
        if match:
            snap_num = int(match.group(1))

        return {
            "MRL":             float(row.get("mrl", float("nan"))),
            "Pairwise":        float(row.get("pairwise", float("nan"))),
            "Hemi_Contrast":   abs(float(row.get("hemi_contrast", float("nan")))),
            "Centroid_Offset": float(centroid_val),
            "N_sat":           n_sat,
            "Group_R_Crit200": r200,
            "snap_num":        snap_num,
            "metrics_file_path": metrics_file,
        }

    def _get_radii_from_positions(
        self, files: list, r200: float, top10: bool = False
    ) -> np.ndarray | None:
        """
        Read a positions CSV and compute satellite distances in units of R_200c.

        Row 0 in all positions files is the central subhalo and is always
        skipped.

        Parameters
        ----------
        files : list[str]
        r200 : float
            Group_R_Crit200 in simulation units.
        top10 : bool
            If True, read ``*_positions_top_10.csv`` (excluding dark files).
            If False (default), read ``*_positions.csv`` (excluding dark and
            top-10 files).

        Returns
        -------
        np.ndarray or None
            Shape ``(N_sat,)`` of float32 values ``r_i / R_200c``.
        """
        if r200 <= 0.0:
            return None

        if top10:
            pos_file = next(
                (f for f in files
                 if "positions_top_10" in os.path.basename(f).lower()
                 and "dark" not in os.path.basename(f).lower()
                 and f.endswith(".csv")),
                None,
            )
        else:
            pos_file = next(
                (f for f in files
                 if "positions" in os.path.basename(f).lower()
                 and "dark" not in os.path.basename(f).lower()
                 and "top_10" not in os.path.basename(f).lower()
                 and f.endswith(".csv")),
                None,
            )

        if pos_file is None or not os.path.exists(pos_file):
            return None

        try:
            df = pd.read_csv(pos_file)
            if df.empty or len(df) < 2:
                return None
            # Row 0 = central subhalo; skip it
            sat_rows = df.iloc[1:]
            if not all(c in sat_rows.columns for c in ("x", "y", "z")):
                return None
            xyz = sat_rows[["x", "y", "z"]].values.astype(np.float64)
            # xyz are relative positions (host at origin)
            radii = np.linalg.norm(xyz, axis=1) / r200
            return radii.astype(np.float32)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # p-value tab update — Single Halo mode
    # ------------------------------------------------------------------

    def _gate_top10_checkboxes(self, n_sat: int) -> None:
        """
        Enable or disable all Top-10 checkboxes based on the halo's
        luminous satellite count.

        When ``n_sat < 10`` the top-10 restriction is meaningless (there
        are not enough satellites to form a genuine top-10 population), so
        all three checkboxes are unchecked, disabled, and given an
        informative tooltip.  When ``n_sat >= 10`` they are re-enabled.

        Parameters
        ----------
        n_sat : int
            Number of luminous satellites (``visible_subhalos`` from the
            full metrics file, **not** capped at 10).
        """
        has_ten = (n_sat is not None and n_sat >= 10)
        tip_off = (
            f"Disabled: this system only has {n_sat} luminous satellites "
            "(requires N \u2265 10)."
        )
        tip_on = "Restrict analysis to the top 10 most massive satellites."

        for chk in (
            self.toggle_top10,
            self.p_value_tab.chk_top10,
            self.sat_centric_tab.chk_top10,
        ):
            if not has_ten:
                chk.blockSignals(True)
                chk.setChecked(False)
                chk.blockSignals(False)
                chk.setEnabled(False)
                chk.setToolTip(tip_off)
            else:
                chk.setEnabled(True)
                chk.setToolTip(tip_on)

    # ------------------------------------------------------------------
    # Lazy evaluation handlers for p-value tabs
    # ------------------------------------------------------------------

    def _on_plot_tab_changed(self, index: int) -> None:
        """
        Lazy evaluation handler triggered when the active plot tab changes.
        Computes p-value tabs on-demand if marked dirty.
        """
        current_widget = self.plot_tabs.widget(index)
        if current_widget == self.p_value_tab and getattr(self, "_p_value_dirty", False):
            self._compute_p_value_tab_if_needed()
        elif current_widget == self.sat_centric_tab and getattr(self, "_sat_centric_dirty", False):
            self._compute_sat_centric_tab_if_needed()

    def _compute_p_value_tab_if_needed(self) -> None:
        """Compute Tab 7 (Statistical Significance) if marked dirty."""
        if self.mode_single_halo.isChecked():
            if hasattr(self, "_cached_p_val_files") and self._cached_p_val_files:
                self.update_p_value_tab_single(self._cached_p_val_files)
        elif self.mode_tracked_history.isChecked():
            if hasattr(self, "current_lineage") and self.current_lineage:
                self.update_p_value_tab_tracked()
        self._p_value_dirty = False

    def _compute_sat_centric_tab_if_needed(self) -> None:
        """Compute Tab 8 (Sat-Centric Significance Evolution) if marked dirty."""
        if self.mode_single_halo.isChecked():
            if hasattr(self, "_cached_p_val_files") and self._cached_p_val_files:
                self.update_sat_centric_tab(self._cached_p_val_files)
        self._sat_centric_dirty = False

    def update_p_value_tab_single(self, files: list) -> None:
        """
        Compute and display p-values in the Statistical Significance tab
        for **Single Halo View** mode.

        Parameters
        ----------
        files : list[str]
            All CSV files in the currently selected halo folder.
        """
        if self.p_value_engine is None:
            self._init_p_value_engine()

        is_top10 = self.p_value_tab.chk_top10.isChecked()

        metrics = self._get_metrics_from_files(files, top10=is_top10)
        if metrics is None:
            self.p_value_tab.canvas.clear_and_show_message(
                "No metrics file found for this halo."
            )
            self.p_value_tab.lbl_status.setText("No metrics file found.")
            return

        n_sat = metrics["N_sat"]
        # Gate all Top-10 checkboxes: disable + uncheck when halo has < 10 satellites
        self._gate_top10_checkboxes(n_sat)
        # Re-read is_top10 after gating (checkbox may have been force-unchecked)
        is_top10 = self.p_value_tab.chk_top10.isChecked()

        if n_sat < 2:
            self.p_value_tab.show_insufficient_data(n_sat)
            return

        observed = {
            "MRL":             metrics["MRL"],
            "Pairwise":        metrics["Pairwise"],
            "Hemi_Contrast":   metrics["Hemi_Contrast"],
            "Centroid_Offset": metrics["Centroid_Offset"],
        }

        # Radii for D_cen on-the-fly computation (always from full positions for R_200c)
        radii = self._get_radii_from_positions(files, metrics["Group_R_Crit200"], top10=is_top10)

        # --- p_iso ---
        p_iso, was_generated = compute_p_iso(
            self.p_value_engine, observed, n_sat, radii_r200=radii
        )

        # Collect null distributions for the histogram canvas
        dist_block, _ = self.p_value_engine.get_or_generate(n_sat)
        null_dists = {
            "R_bar":     dist_block.get("R_bar"),
            "theta_lop": dist_block.get("theta_lop"),
            "C_hemi":    dist_block.get("C_hemi"),
        }
        # Also attach D_cen null if radii available (computed above inside compute_p_iso)
        if radii is not None and len(radii) >= 2:
            try:
                null_dists["D_cen"] = _compute_d_cen_null(n_sat, radii, n_iter=N_ITER_DEFAULT)
            except Exception:
                null_dists["D_cen"] = None

        # --- p_cosmo ---
        snap_num = metrics.get("snap_num")
        p_cosmo: dict = {}
        rank_info: dict = {}
        if snap_num is not None and self.base_export_folder:
            cosmo_df = build_cosmo_ensemble(self.base_export_folder, snap_num, top10=is_top10)
            if not cosmo_df.empty:
                p_cosmo, rank_info = compute_p_cosmo(observed, cosmo_df)

        # Fill in NaN placeholders for missing p_cosmo
        for k in ("MRL", "Pairwise", "Hemi_Contrast", "Centroid_Offset"):
            p_cosmo.setdefault(k, float("nan"))
            rank_info.setdefault(k, (float("nan"), float("nan")))

        self.p_value_tab.update_for_single_halo(
            observed_values=observed,
            p_iso_values=p_iso,
            p_cosmo_values=p_cosmo,
            rank_info=rank_info,
            null_dists=null_dists,
            n_sat=n_sat,
            from_cache=not was_generated,
        )

    # ------------------------------------------------------------------
    # p-value tab update — Tracked Halo History mode
    # ------------------------------------------------------------------

    def update_p_value_tab_tracked(self) -> None:
        """
        Compute temporal p-value trajectories and display them in the
        Statistical Significance tab for **Tracked Halo History** mode.

        Iterates all snapshots in ``self.current_lineage``, reads metric files,
        computes p_iso + p_cosmo for each, and passes the time series to the
        canvas.
        """
        if self.p_value_engine is None:
            self._init_p_value_engine()

        if not self.current_lineage or not self.base_export_folder:
            self.p_value_tab.canvas.clear_and_show_message("No lineage data loaded.")
            return

        # Determine N_sat at the latest snapshot to gate top-10 checkboxes
        latest_snap = max(self.current_lineage.keys())
        latest_halo_id = self.current_lineage[latest_snap]
        latest_dir = os.path.join(
            self.base_export_folder, f"Snap{latest_snap}", f"Halo {latest_halo_id}"
        )
        if os.path.isdir(latest_dir):
            latest_files = glob.glob(os.path.join(latest_dir, "*.csv"))
            latest_metrics = self._get_metrics_from_files(latest_files, top10=False)
            snap99_n_sat = latest_metrics["N_sat"] if latest_metrics else 0
        else:
            snap99_n_sat = 0
        self._gate_top10_checkboxes(snap99_n_sat)

        # Authoritative top10 state comes from the p-value tab checkbox.
        # Synchronise the sidebar toggle_top10 so both stay in sync.
        is_top10 = self.p_value_tab.chk_top10.isChecked()
        self.toggle_top10.blockSignals(True)
        self.toggle_top10.setChecked(is_top10)
        self.toggle_top10.blockSignals(False)

        snap_nums = []
        p_iso_by_metric:   dict = {k: [] for k in ("MRL", "Pairwise", "Hemi_Contrast", "Centroid_Offset")}
        p_cosmo_by_metric: dict = {k: [] for k in ("MRL", "Pairwise", "Hemi_Contrast", "Centroid_Offset")}

        nan = float("nan")

        with self.busy_state("Computing Tracked P-Values..."):
            sorted_snaps = sorted(self.current_lineage.keys())
            for snap_num in tqdm(
                sorted_snaps,
                desc=f"[P-Value Tracked] Computing p-values (Halo {latest_halo_id})",
                unit="snap",
                leave=False,
            ):
                self.pump_events()
                halo_id = self.current_lineage[snap_num]
                halo_dir = os.path.join(
                    self.base_export_folder, f"Snap{snap_num}", f"Halo {halo_id}"
                )
                if not os.path.isdir(halo_dir):
                    continue

                # Collect ALL CSV files for this snapshot so that
                # _get_metrics_from_files can find both the top-10 file (for
                # metrics) and the full file (for Group_R_Crit200), and
                # _get_radii_from_positions can find the correct positions file.
                snap_files = glob.glob(os.path.join(halo_dir, f"*_snap_{snap_num}_*.csv"))

                metrics = self._get_metrics_from_files(snap_files, top10=is_top10)
                if metrics is None:
                    continue

                n_sat = metrics["N_sat"]
                snap_nums.append(snap_num)

                if n_sat < 2:
                    for k in p_iso_by_metric:
                        p_iso_by_metric[k].append(nan)
                        p_cosmo_by_metric[k].append(nan)
                    continue

                observed = {
                    "MRL":             metrics["MRL"],
                    "Pairwise":        metrics["Pairwise"],
                    "Hemi_Contrast":   metrics["Hemi_Contrast"],
                    "Centroid_Offset": metrics["Centroid_Offset"],
                }
                radii = self._get_radii_from_positions(snap_files, metrics["Group_R_Crit200"], top10=is_top10)

                # p_iso (cache hit for previously seen N_sat values)
                try:
                    iso_vals, _ = compute_p_iso(
                        self.p_value_engine, observed, n_sat, radii_r200=radii
                    )
                except Exception:
                    iso_vals = {k: nan for k in p_iso_by_metric}

                for k in p_iso_by_metric:
                    p_iso_by_metric[k].append(iso_vals.get(k, nan))

                # p_cosmo at this specific snapshot's ensemble (respects top10 flag)
                cosmo_df = build_cosmo_ensemble(self.base_export_folder, snap_num, top10=is_top10)
                if cosmo_df.empty:
                    for k in p_cosmo_by_metric:
                        p_cosmo_by_metric[k].append(nan)
                else:
                    try:
                        cosmo_vals, _ = compute_p_cosmo(observed, cosmo_df)
                    except Exception:
                        cosmo_vals = {k: nan for k in p_cosmo_by_metric}
                    for k in p_cosmo_by_metric:
                        p_cosmo_by_metric[k].append(cosmo_vals.get(k, nan))

        if not snap_nums:
            self.p_value_tab.canvas.clear_and_show_message(
                "No metric data found along this lineage."
            )
            return

        self.p_value_tab.update_for_tracked_history(
            snap_nums, p_iso_by_metric, p_cosmo_by_metric
        )

    # ------------------------------------------------------------------
    # Force recalculate button handler
    # ------------------------------------------------------------------

    def _on_pvalue_recalculate(self) -> None:
        """
        Triggered by the 'Recalculate / Force Re-seed Cache' button.

        In Single Halo mode: clears the cache entry for the current N_sat
        and recomputes from scratch.
        In Tracked History mode: re-runs the full temporal computation
        (individual N_sat entries are only cleared if they appear in the
        lineage, to avoid discarding unrelated cached distributions).
        """
        if self.p_value_engine is None:
            self._init_p_value_engine()

        is_single  = self.mode_single_halo.isChecked()
        is_tracked = self.mode_tracked_history.isChecked()
        is_top10   = self.p_value_tab.chk_top10.isChecked()

        with self.busy_state("Recalculating P-Values..."):
            if is_single:
                if not hasattr(self, "_current_halo_dir") or not self._current_halo_dir:
                    return
                files = [
                    os.path.join(self._current_halo_dir, f)
                    for f in os.listdir(self._current_halo_dir)
                    if f.endswith(".csv")
                ]
                metrics = self._get_metrics_from_files(files, top10=is_top10)
                if metrics and metrics["N_sat"] >= 2:
                    self.p_value_engine.clear_entry(metrics["N_sat"])
                    invalidate_cosmo_ensemble_cache()
                self.update_p_value_tab_single(files)
                self._p_value_dirty = False

            elif is_tracked:
                # Clear cache entries for all N_sat values seen in this lineage
                if self.current_lineage and self.base_export_folder:
                    invalidate_cosmo_ensemble_cache()
                    for snap_num, halo_id in tqdm(
                        self.current_lineage.items(),
                        desc="[P-Value] Resetting lineage cache",
                        unit="snap",
                        leave=False,
                    ):
                        self.pump_events()
                        halo_dir = os.path.join(
                            self.base_export_folder, f"Snap{snap_num}", f"Halo {halo_id}"
                        )
                        # Gather all CSVs so _get_metrics_from_files can resolve
                        # Group_R_Crit200 from the full file even in top10 mode
                        snap_files = glob.glob(os.path.join(halo_dir, f"*_snap_{snap_num}_*.csv"))
                        metrics = self._get_metrics_from_files(snap_files, top10=is_top10)
                        if metrics and metrics["N_sat"] >= 2:
                            self.p_value_engine.clear_entry(metrics["N_sat"])
                self.update_p_value_tab_tracked()
                self._p_value_dirty = False

    def export_p_value_plot_light(self) -> None:
        """
        Save the current Statistical Significance canvas to a PNG file with a
        white background (light theme), matching the export behaviour of the
        Metric Evolution tab.
        """
        # Determine a sensible default filename based on the current mode
        if self.mode_single_halo.isChecked():
            default_name = "Statistical_Significance_Single.png"
        elif self.mode_tracked_history.isChecked():
            default_name = "Statistical_Significance_Temporal.png"
        else:
            default_name = "Statistical_Significance.png"

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Statistical Significance Plot (Light BG)",
            default_name,
            "PNG Image (*.png);;All Files (*)",
        )
        if filepath:
            try:
                self.p_value_tab.canvas.export_white_bg(filepath)
                QMessageBox.information(
                    self, "Export Successful",
                    f"Plot exported to:\n{filepath}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export plot:\n{e}")

    # ------------------------------------------------------------------
    # Top-10 toggle handlers for p-value and sat-centric tabs
    # ------------------------------------------------------------------

    def _on_pvalue_top10_toggled(self) -> None:
        """Re-run the p-value tab update when the Top-10 checkbox is toggled."""
        invalidate_cosmo_ensemble_cache()
        if self.mode_single_halo.isChecked():
            # Re-use the file list already populated in the current view
            files = [
                os.path.join(self._current_halo_dir, f)
                for f in os.listdir(self._current_halo_dir)
                if f.endswith(".csv")
            ] if hasattr(self, "_current_halo_dir") and self._current_halo_dir else []
            if files:
                self.update_p_value_tab_single(files)
                self._p_value_dirty = False
        elif self.mode_tracked_history.isChecked():
            self.update_p_value_tab_tracked()
            self._p_value_dirty = False

    # ------------------------------------------------------------------
    # Satellite-Centric Significance Evolution tab
    # ------------------------------------------------------------------

    def update_sat_centric_tab(self, files: list) -> None:
        """
        Populate the Satellite-Centric Significance Evolution tab for a
        single halo in **Single Halo View** mode.

        Reads the ``*_metric_evolution.csv`` (or top-10 variant), iterates
        over all snapshot rows, and computes p_iso + p_cosmo at each epoch
        using the **fixed** Snap-99 satellite multiplicity and radii.

        Parameters
        ----------
        files : list[str]
            All CSV files in the currently selected halo folder.
        """
        if self.p_value_engine is None:
            self._init_p_value_engine()

        is_top10 = self.sat_centric_tab.chk_top10.isChecked()

        # 1. Find the metric_evolution file
        evol_suffix = "_metric_evolution_top_10.csv" if is_top10 else "_metric_evolution.csv"
        evol_file = next(
            (f for f in files
             if os.path.basename(f).endswith(evol_suffix)
             and "top_10" not in os.path.basename(f).replace(evol_suffix, "")),
            None,
        )
        # Fallback: any metric_evolution file
        if evol_file is None:
            evol_file = next(
                (f for f in files if "metric_evolution" in os.path.basename(f)
                 and f.endswith(".csv")),
                None,
            )
        if evol_file is None or not os.path.exists(evol_file):
            self.sat_centric_tab.show_message("No metric evolution file found for this halo.")
            return

        # 2. Read Snap-99 metrics for fixed N_sat and R_200c
        # Always use the full metrics file for N_sat (for gating) and R_200c
        snap99_metrics_full = self._get_metrics_from_files(files, top10=False)
        n_sat_full = snap99_metrics_full["N_sat"] if snap99_metrics_full else 0
        # Gate checkboxes based on the full luminous count
        self._gate_top10_checkboxes(n_sat_full)
        # Re-read is_top10 after gating (checkbox may have been unchecked)
        is_top10 = self.sat_centric_tab.chk_top10.isChecked()

        snap99_metrics = self._get_metrics_from_files(files, top10=is_top10)
        if snap99_metrics is None or snap99_metrics["N_sat"] < 2:
            n = snap99_metrics["N_sat"] if snap99_metrics else 0
            self.sat_centric_tab.show_message(
                f"N_sat = {n}: angular statistics require \u2265 2 satellites."
            )
            return

        n_sat = snap99_metrics["N_sat"]
        r200  = snap99_metrics["Group_R_Crit200"]

        # 3. Fixed radii from Snap-99 positions (top-10 positions when is_top10)
        radii = self._get_radii_from_positions(files, r200, top10=is_top10)

        # 4. Read evolution DataFrame
        try:
            df_evol = pd.read_csv(evol_file)
        except Exception as exc:
            self.sat_centric_tab.show_message(f"Failed to read evolution file: {exc}")
            return

        # Normalise column names
        col_rename = {
            "Centroid_Distance": "Centroid_Offset",
            "centroid":          "Centroid_Offset",
            "MRL":               "MRL",
            "Pairwise":          "Pairwise",
            "Hemi_Contrast":     "Hemi_Contrast",
        }
        df_evol = df_evol.rename(columns=col_rename)

        if "SnapNum" not in df_evol.columns:
            self.sat_centric_tab.show_message("Evolution file missing 'SnapNum' column.")
            return

        # 5. Iterate snapshots and collect p-values
        nan = float("nan")
        snap_nums: list = []
        p_iso_by_metric:   dict = {k: [] for k in ("MRL", "Pairwise", "Hemi_Contrast", "Centroid_Offset")}
        p_cosmo_by_metric: dict = {k: [] for k in ("MRL", "Pairwise", "Hemi_Contrast", "Centroid_Offset")}

        with self.busy_state("Evaluating Satellite-Centric Trajectories..."):
            for _, row in tqdm(
                df_evol.iterrows(),
                total=len(df_evol),
                desc="[P-Value Sat-Centric] Evaluating trajectory",
                unit="snap",
                leave=False,
            ):
                self.pump_events()
                snap_num = int(row["SnapNum"])
                snap_nums.append(snap_num)

                observed = {}
                for k in ("MRL", "Pairwise", "Hemi_Contrast", "Centroid_Offset"):
                    observed[k] = float(row.get(k, nan))

                # p_iso — use fixed N_sat from Snap 99
                try:
                    iso_vals, _ = compute_p_iso(
                        self.p_value_engine, observed, n_sat, radii_r200=radii
                    )
                except Exception:
                    iso_vals = {k: nan for k in p_iso_by_metric}

                for k in p_iso_by_metric:
                    p_iso_by_metric[k].append(iso_vals.get(k, nan))

                # p_cosmo — compare against snapshot ensemble
                if self.base_export_folder:
                    try:
                        cosmo_df = build_cosmo_ensemble(
                            self.base_export_folder, snap_num, top10=is_top10
                        )
                        if not cosmo_df.empty:
                            cosmo_vals, _ = compute_p_cosmo(observed, cosmo_df)
                        else:
                            cosmo_vals = {k: nan for k in p_cosmo_by_metric}
                    except Exception:
                        cosmo_vals = {k: nan for k in p_cosmo_by_metric}
                else:
                    cosmo_vals = {k: nan for k in p_cosmo_by_metric}

                for k in p_cosmo_by_metric:
                    p_cosmo_by_metric[k].append(cosmo_vals.get(k, nan))

        if not snap_nums:
            self.sat_centric_tab.show_message("No snapshot data found in evolution file.")
            return

        mode_str = "Top-10" if is_top10 else "Full"
        status = f"N_sat={n_sat}  |  {mode_str} satellites  |  {len(snap_nums)} snapshots"
        self.sat_centric_tab.update_plot(snap_nums, p_iso_by_metric, p_cosmo_by_metric, status)

    def _on_sat_centric_recalculate(self) -> None:
        """Force-recalculate button for the Satellite-Centric tab."""
        if not hasattr(self, "_current_halo_dir") or not self._current_halo_dir:
            return
        files = [
            os.path.join(self._current_halo_dir, f)
            for f in os.listdir(self._current_halo_dir)
            if f.endswith(".csv")
        ]
        if files:
            with self.busy_state("Recalculating P-Values..."):
                self.update_sat_centric_tab(files)
                self._sat_centric_dirty = False

    def _on_sat_centric_top10_toggled(self) -> None:
        """Re-run sat-centric tab when its Top-10 checkbox is toggled."""
        invalidate_cosmo_ensemble_cache()
        self._on_sat_centric_recalculate()

    def export_sat_centric_plot_light(self) -> None:
        """Save the Satellite-Centric canvas to PNG with white background."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Sat-Centric Significance Plot (Light BG)",
            "SatCentric_Significance_Evolution.png",
            "PNG Image (*.png);;All Files (*)",
        )
        if filepath:
            try:
                self.sat_centric_tab.canvas.export_white_bg(filepath)
                QMessageBox.information(
                    self, "Export Successful", f"Plot exported to:\n{filepath}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export plot:\n{e}")

    def read_single_snap_metrics(self, metrics_file_path):
        """
        Loads metrics and returns the 3D plot context (mass, R_Crit200).
        This method replaces the logic extracted from load_and_display_metrics/update_3d_view_for_lineage_snap.
        """
        central_mass = 50.0
        group_r_crit200 = 0.0

        if not metrics_file_path:
            return central_mass, group_r_crit200

        try:
            # The metrics file is expected to be a single-row CSV
            if os.path.exists(metrics_file_path):
                metrics_data = pd.read_csv(metrics_file_path).iloc[0].to_dict()

                # Backward-compat: rename legacy key before display
                if 'Centroid_Distance' in metrics_data and 'Centroid_Offset' not in metrics_data:
                    metrics_data['Centroid_Offset'] = metrics_data.pop('Centroid_Distance')

                # Format the data for display in the QTextEdit
                metrics_display = ""
                for key, value in metrics_data.items():
                    if isinstance(value, float):
                        metrics_display += f"{key}: {value:.4f}\n"
                    elif isinstance(value, np.ndarray):
                        metrics_display += f"{key}: {np.array2string(value, separator=', ', precision=2)}\n"
                    else:
                        metrics_display += f"{key}: {value}\n"

                self.metrics_text.setText(metrics_display)

                # Extract context for 3D plot
                try:
                    central_mass = float(metrics_data.get("Central_Galaxy_Mass", 50.0))
                except (ValueError, TypeError):
                    central_mass = 50.0

                try:
                    group_r_crit200 = float(metrics_data.get("Group_R_Crit200", 0.0))
                except (ValueError, TypeError):
                    group_r_crit200 = 0.0
            else:
                self.metrics_text.setText("Metrics file not found for this snapshot.")

        except FileNotFoundError:
            self.metrics_text.setText("Metrics file not found for this snapshot.")
        except Exception as e:
            self.metrics_text.setText(f"Error loading metrics: {e}")

        # Fallback Retrieval of Group_R_Crit200:
        # If group_r_crit200 <= 0.0, look for corresponding non-top-10 metrics CSV file
        if group_r_crit200 <= 0.0 and metrics_file_path:
            folder = os.path.dirname(metrics_file_path)
            full_metrics_path = None
            if "top_10" in os.path.basename(metrics_file_path):
                cand = metrics_file_path.replace("_metrics_top_10.csv", "_metrics.csv")
                if os.path.exists(cand):
                    full_metrics_path = cand
            if not full_metrics_path and os.path.isdir(folder):
                candidates = [
                    f for f in glob.glob(os.path.join(folder, "*_metrics.csv"))
                    if "top_10" not in os.path.basename(f)
                ]
                if candidates:
                    full_metrics_path = candidates[0]

            if full_metrics_path and os.path.exists(full_metrics_path):
                try:
                    df_full = pd.read_csv(full_metrics_path)
                    if not df_full.empty:
                        row = df_full.iloc[0]
                        group_r_crit200 = float(row.get("Group_R_Crit200", 0.0))
                        if central_mass == 50.0 and "Central_Galaxy_Mass" in row:
                            try:
                                central_mass = float(row.get("Central_Galaxy_Mass", 50.0))
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    pass

        return central_mass, group_r_crit200


    def gather_lineage_metrics(self):
        # Clears and gathers the single-halo (Host-Centric) metrics for the selected lineage
        self.lineage_metrics = {
            "SnapNum": [], "Centroid_Offset": [], "MRL": [], "Pairwise": [],
            "Hemi_Contrast": [], "Normalised_RMS_Distance": []
        }

        # Define the keys we want in our internal dict (Capitalized)
        target_keys = ["Centroid_Offset", "MRL", "Pairwise", "Hemi_Contrast", "Normalised_RMS_Distance"]

        # Define possible column names in the CSV for each target key (Capitalized and Lowercase)
        key_map = {
            "Centroid_Offset": ["Centroid_Offset", "Centroid_Distance", "centroid"],
            "MRL": ["MRL", "mrl"],
            "Pairwise": ["Pairwise", "pairwise"],
            "Hemi_Contrast": ["Hemi_Contrast", "hemi_contrast"],
            "Normalised_RMS_Distance": ["Normalised_RMS_Distance", "normalised_rms_distance"]
        }

        is_top10 = self.toggle_top10.isChecked()

        # Iterate through the lineage dictionary (SnapNum: HaloID)
        for snap, halo_id in sorted(self.current_lineage.items()):

            # FLAT STRUCTURE PATH
            parent_dir = os.path.join(self.base_export_folder, f"Snap{snap}", f"Halo {halo_id}")

            # Choose file based on checkbox and check for ANY simulation prefix
            if is_top10:
                # WILDCARD SEARCH FOR SIMULATION PREFIX
                pattern = os.path.join(parent_dir, f"sim_*_halo_{halo_id}_snap_{snap}_metrics_top_10.csv")
            else:
                pattern = os.path.join(parent_dir, f"sim_*_halo_{halo_id}_snap_{snap}_metrics.csv")

            candidates = glob.glob(pattern)

            if candidates:
                metric_file = candidates[0] # Take the first match
                try:
                    df = pd.read_csv(metric_file)

                    # Check if we can find a valid column for every required metric
                    row_values = {}
                    all_found = True

                    for target in target_keys:
                        found_col = None
                        # check possible csv headers
                        for candidate in key_map[target]:
                            if candidate in df.columns:
                                found_col = candidate
                                break

                        if found_col:
                            row_values[target] = df[found_col].iloc[0]
                        else:
                            all_found = False
                            break

                    if all_found:
                        self.lineage_metrics["SnapNum"].append(snap)
                        for target in target_keys:
                            self.lineage_metrics[target].append(row_values[target])
                    else:
                        pass

                except Exception as e:
                    pass
            else:
                # If top 10 requested but file doesn't exist, simply skip adding to metrics.
                pass


    def create_3d_plot_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        canvas = Plot3DCanvas(self, tab)
        toolbar = NavigationToolbar(canvas, self)
        layout.addWidget(canvas)
        layout.addWidget(toolbar)

        export_3d_button = QPushButton("Export 3D View (White BG)")
        export_3d_button.clicked.connect(self.export_3d_view)
        layout.addWidget(export_3d_button)

        return tab, canvas, toolbar

    def export_3d_view(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export 3D View",
            "3D_View_Export.png",
            "PNG Image (*.png);;All Files (*)"
        )
        if filepath:
            try:
                self.canvas_3d.export_projections_white_bg(filepath)
                base, ext = os.path.splitext(filepath)
                if not ext:
                    ext = ".png"
                top_down = f"{os.path.basename(base)}_XY_TopDown{ext}"
                side_on = f"{os.path.basename(base)}_XZ_SideOn{ext}"
                main_file = os.path.basename(filepath)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"3D views exported successfully:\n- {main_file}\n- {top_down}\n- {side_on}\n\nSaved to folder:\n{os.path.dirname(filepath)}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export 3D View: {e}")


    def create_metric_plot_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        canvas = MetricEvolutionCanvas(self)
        toolbar = NavigationToolbar(canvas, self)
        layout.addWidget(canvas)
        layout.addWidget(toolbar)

        export_layout = QHBoxLayout()
        btn_export_core = QPushButton("Export Core Metrics (2x2 Light)")
        btn_export_all = QPushButton("Export All Metrics (Light)")
        export_layout.addWidget(btn_export_core)
        export_layout.addWidget(btn_export_all)
        layout.addLayout(export_layout)

        return tab, canvas, toolbar, btn_export_core, btn_export_all

    def export_lineage_core(self):
        if not hasattr(self.canvas_metrics, 'current_data') or not self.canvas_metrics.current_data:
            QMessageBox.warning(self, "Export Failed", "No metric evolution data available to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Core Metrics (2x2)",
            "Core_Metric_Evolution.png",
            "PNG Image (*.png);;All Files (*)"
        )
        if filepath:
            try:
                self.canvas_metrics.export_core_light(filepath)
                QMessageBox.information(self, "Export Successful", f"Core metrics exported to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export core metrics: {e}")

    def export_lineage_all(self):
        if not hasattr(self.canvas_metrics, 'current_data') or not self.canvas_metrics.current_data:
            QMessageBox.warning(self, "Export Failed", "No metric evolution data available to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export All Metrics",
            "Lineage_Metric_Evolution.png",
            "PNG Image (*.png);;All Files (*)"
        )
        if filepath:
            try:
                self.canvas_metrics.export_all_light(filepath)
                QMessageBox.information(self, "Export Successful", f"All metrics exported to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export all metrics: {e}")

    def export_overview_satellite_white_bg(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Satellite-Centric Overview",
            "Overview_Satellite_Centric_White.png",
            "PNG Image (*.png);;All Files (*)"
        )
        if filepath:
            try:
                self.canvas_overview_satellite.export_overview_white_bg(filepath)
                QMessageBox.information(self, "Export Successful", f"Saved satellite-centric overview to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export overview: {e}")

    def export_overview_host_white_bg(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Host-Centric Overview",
            "Overview_Host_Centric_White.png",
            "PNG Image (*.png);;All Files (*)"
        )
        if filepath:
            try:
                self.canvas_overview_host.export_overview_white_bg(filepath)
                QMessageBox.information(self, "Export Successful", f"Saved host-centric overview to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export overview: {e}")

    def export_corner_plot_white_bg(self):
        """Exports the Snapshot 99 Corner Plot with a clean white background."""
        if not hasattr(self, 'snap99_base_df') or self.snap99_base_df.empty:
            QMessageBox.warning(self, "Export Failed", "No corner plot data available to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Corner Plot",
            "Corner_Plot_Snap99_White.png",
            "PNG Image (*.png);;All Files (*)"
        )
        if not filepath:
            return

        total_halos = len(self.snap99_base_df)
        core_metrics = ["Centroid_Offset", "MRL", "Pairwise", "Hemi_Contrast"]

        selected = self.corner_category_combo.currentText()
        top_ids = set()
        bottom_ids = set()

        if selected == "Majority Vote (>= 3 Metrics)" or not self.system_table_data:
            for halo_id in self.snap99_base_df['HaloID']:
                top_count = sum(
                    halo_id in self.system_table_data.get(m, {}).get('Top 1/3', {})
                    for m in core_metrics
                )
                bottom_count = sum(
                    halo_id in self.system_table_data.get(m, {}).get('Bottom 1/3', {})
                    for m in core_metrics
                )
                if top_count >= 3:
                    top_ids.add(halo_id)
                elif bottom_count >= 3:
                    bottom_ids.add(halo_id)
        else:
            metric_data = self.system_table_data.get(selected, {})
            top_ids = set(metric_data.get('Top 1/3', {}).keys())
            bottom_ids = set(metric_data.get('Bottom 1/3', {}).keys())

        middle_ids = set(self.snap99_base_df['HaloID']) - top_ids - bottom_ids
        if total_halos > 0:
            lbl_top = f"Most Lopsided ({len(top_ids)} | {len(top_ids) / total_halos * 100:.1f}%)"
            lbl_bot = f"Least Lopsided ({len(bottom_ids)} | {len(bottom_ids) / total_halos * 100:.1f}%)"
            lbl_mid = f"Remaining Systems ({len(middle_ids)} | {len(middle_ids) / total_halos * 100:.1f}%)"
        else:
            lbl_top = "Most Lopsided"
            lbl_bot = "Least Lopsided"
            lbl_mid = "Remaining Systems (0 | 0.0%)"

        def assign_category(hid):
            if hid in top_ids:
                return lbl_top
            if hid in bottom_ids:
                return lbl_bot
            return lbl_mid

        df = self.snap99_base_df.copy()
        df['Category'] = df['HaloID'].map(assign_category)

        ordered_metrics = [
            self.corner_metric_list.item(i).text()
            for i in range(self.corner_metric_list.count())
        ]

        if self.corner_pct_checkbox.isChecked():
            for metric in ordered_metrics:
                if metric in df.columns:
                    df[metric] = df[metric].rank(pct=True)

        if df.empty or not ordered_metrics:
            QMessageBox.warning(self, "Export Failed", "No metric data available to export.")
            return

        with plt.style.context("default"):
            grid = sns.pairplot(
                df[ordered_metrics + ['Category']],
                vars=ordered_metrics,
                hue='Category',
                palette={
                    lbl_top: 'red',
                    lbl_bot: 'blue',
                    lbl_mid: 'gray',
                },
                corner=True,
                plot_kws={'alpha': 0.6},
            )
            grid.figure.patch.set_facecolor('white')
            grid.figure.set_facecolor('white')
            for ax in grid.axes.flatten():
                if ax is not None:
                    ax.set_facecolor('white')
                    ax.tick_params(colors='black')
                    for spine in ax.spines.values():
                        spine.set_color('black')
            try:
                grid.figure.savefig(filepath, facecolor='white', dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Export Successful", f"Saved corner plot to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export corner plot: {e}")
            finally:
                plt.close(grid.figure)


    def update_ui_for_mode(self):
        is_single = self.mode_single_halo.isChecked()
        is_tracked = self.mode_tracked_history.isChecked()
        is_overview = self.mode_metrics_overview.isChecked()
        is_reasons = self.mode_potential_reasons.isChecked()

        # Update controls
        self.single_halo_group.setVisible(is_single)
        self.history_group.setVisible(is_tracked)
        self.overview_group.setVisible(is_overview)
        self.reasons_group.setVisible(is_reasons)

        # Visibility for common elements
        self.metrics_label.setVisible(is_single or is_tracked)
        self.metrics_text.setVisible(is_single or is_tracked)
        self.toggle_dark.setVisible(is_single or is_tracked)
        self.toggle_tracks.setVisible(is_single or is_tracked)
        self.toggle_box.setVisible(is_single or is_tracked)

        # Enable Top 10 Toggle for Single, Tracked, AND Overview modes
        self.toggle_top10.setVisible(is_single or is_tracked or is_overview)

        # Plot tabs visibility
        # 0: 3D View, 1: Metric History, 2: Satellite-Centric Overview,
        # 3: Host-Centric Overview, 4: Systems Table, 5: Corner Plot,
        # 6: Potential Reasons, 7: Statistical Significance (p-values)
        self.plot_tabs.setTabVisible(0, is_single or is_tracked)
        self.plot_tabs.setTabVisible(1, is_single or is_tracked)
        self.plot_tabs.setTabVisible(2, is_overview)
        self.plot_tabs.setTabVisible(3, is_overview)
        self.plot_tabs.setTabVisible(4, is_overview)
        self.plot_tabs.setTabVisible(5, is_overview)
        self.plot_tabs.setTabVisible(6, is_reasons)
        self.plot_tabs.setTabVisible(7, is_single or is_tracked)
        self.plot_tabs.setTabVisible(8, is_single)   # Sat-Centric: Single Halo only

        if is_overview:
            self.plot_tabs.setCurrentIndex(2)
        elif is_tracked:
            self.plot_tabs.setCurrentIndex(0) # Reset to 3D view when tracking
        elif is_reasons:
            self.plot_tabs.setCurrentIndex(6)

        self.clear_all()


    def select_folder(self, prompt_text="Select Folder"):
        folder = QFileDialog.getExistingDirectory(self, prompt_text)
        if not folder: return

        # Determine if the selected folder is a specific Halo folder or the base export folder
        is_specific_halo_folder = "Halo" in os.path.basename(folder)

        if self.mode_single_halo.isChecked():
            # Single Halo selection must point directly to SnapXX/Halo ID
            if not is_specific_halo_folder:
                 QMessageBox.warning(self, "Invalid Selection", "Please select a specific 'Halo ID' folder (e.g., .../Snap99/Halo 488).")
                 return

            # The base export folder is now two levels up: Base/SnapXX/Halo ID
            self.base_export_folder = os.path.dirname(os.path.dirname(folder))
            self._init_p_value_engine()   # rebind engine to new folder's .p_val_cache
            self.current_halo_folder = folder # Store the specific folder
            self.file_list.clear()
            files = glob.glob(os.path.join(folder, "*.csv"))
            self.file_list.addItems([os.path.basename(f) for f in files])
            self.update_current_view()

        elif self.mode_tracked_history.isChecked() or self.mode_metrics_overview.isChecked():
            # For tracked history or overview, we need the base export folder
            self.base_export_folder = folder
            self._init_p_value_engine()   # rebind engine to new folder's .p_val_cache
            if self.mode_tracked_history.isChecked():
                self.load_lineage_summary()
            elif self.mode_metrics_overview.isChecked():
                self.plot_metrics_overview()

            # --- INTELLIGENT LOADING (Auto-detect child folder) ---
            # Try to find 'merger_plots' inside this folder
            merger_path = os.path.join(self.base_export_folder, "merger_plots")
            if os.path.exists(merger_path):
                self.view_potential_reasons.load_data(merger_path)
                self.lbl_reasons_status.setText(f"Auto-Loaded: merger_plots")


    def select_reasons_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Merger History Folder")
        if folder:
            self.lbl_reasons_status.setText(f"Loaded: {os.path.basename(folder)}")
            self.view_potential_reasons.load_data(folder)

            # --- INTELLIGENT LOADING (Auto-detect parent folder) ---
            # If we haven't loaded the main history yet, check the parent dir
            if not self.base_export_folder:
                parent_dir = os.path.dirname(folder)
                lineage_file = os.path.join(parent_dir, "halo_lineage_summary.csv")

                if os.path.exists(lineage_file):
                    self.base_export_folder = parent_dir
                    # We can preload the lineage summary silently or explicitly
                    self.load_lineage_summary()
                    print(f"Auto-detected Base Export Folder: {parent_dir}")


    def load_and_display_metrics(self, metrics_file):
        """Loads metrics from a single-snapshot file and displays them."""
        # This function is now OBSOLETE/deprecated, replaced by read_single_snap_metrics.
        # It's kept here just in case, but its usage should be replaced.
        return self.read_single_snap_metrics(metrics_file)

    
    # Aggregation and Plotting for All Systems (Quantile Style)
    def plot_metrics_overview(self):
        if not self.base_export_folder:
            QMessageBox.warning(self, "No Folder Selected", "Please select the base 'halo_exports' folder.")
            return

        is_top10 = self.toggle_top10.isChecked()

        # Load Top 10 Reference List if active
        valid_top10_map = {} # {snap_num: set(halo_ids)}
        if is_top10:
            top10_ref_file = os.path.join(self.base_export_folder, "top_10_systems.csv")
            if not os.path.exists(top10_ref_file):
                # Fallback check for summary file
                top10_ref_file = os.path.join(self.base_export_folder, "top_10_systems_summary.csv")

            if os.path.exists(top10_ref_file):
                try:
                    df_top10 = pd.read_csv(top10_ref_file)
                    valid_top10_map = parse_top10_reference(df_top10)
                except Exception as e:
                    print(f"Error parsing top_10_systems.csv: {e}")
                    QMessageBox.warning(self, "Reference Error", "Could not parse top_10_systems.csv. Filtering disabled.")
            else:
                QMessageBox.warning(self, "File Missing", "top_10_systems.csv not found in folder. Run generate_top10_list.py first.")
                return

        # Use Standard Metrics
        metrics_keys = ["Centroid_Offset", "MRL", "Pairwise", "Hemi_Contrast", "Normalised_RMS_Distance"]
        metrics_csv_keys = ['centroid', 'mrl', 'pairwise', 'hemi_contrast', 'normalised_rms_distance']

        # Table data storage: {MetricKey: {Snap99_ID: MetricValue}}
        self.system_table_data = {}

        # Store sorted IDs for the "Irresponsible Plot" export feature
        self.sorted_ids_cache = {'Sat': [], 'Host': []}

        snap_folders = sorted(glob.glob(os.path.join(self.base_export_folder, "Snap*")))

        if not snap_folders:
            QMessageBox.warning(self, "No Snap Folders", "No 'Snap*' folders found in the selected directory.")
            return

        valid_snaps = [int(os.path.basename(f).replace("Snap", "")) for f in snap_folders]

        # Load the lineage file to map HaloID_SnapXX back to HaloID_Snap99
        lineage_file_path = os.path.join(self.base_export_folder, "halo_lineage_summary.csv")
        if not os.path.exists(lineage_file_path):
             QMessageBox.critical(self, "File Error", "The lineage summary is required for overview plotting.")
             return
        lineage_df = pd.read_csv(lineage_file_path, index_col="HaloID_at_Snap99")
        self.lineage_for_plotting = lineage_df

        with self.busy_state("Loading & Aggregating Overview Data..."):
            # --- Load Satellite-Centric rows (Snap99 fixed progenitors) ---
            sat_snap_num = 99
            sat_file_pattern = "*snap_99_metric_evolution_top_10.csv" if is_top10 else "*snap_99_metric_evolution.csv"
            sat_search_path = os.path.join(self.base_export_folder, f"Snap{sat_snap_num}", "Halo*", sat_file_pattern)
            sat_metric_files = glob.glob(sat_search_path)

            sat_frames = []
            for metric_file in tqdm(sat_metric_files, desc="Loading Sat-Centric"):
                self.pump_events()
                try:
                    df_metric = pd.read_csv(metric_file)
                    if df_metric.empty:
                        continue
                    df_metric = df_metric[df_metric['SnapNum'].isin(valid_snaps)]
                    if df_metric.empty:
                        continue
                    snap99_id = int(re.search(r'_halo_(\d+)_snap_99', metric_file).group(1))
                    df_metric = df_metric.copy()
                    df_metric["snap99_id"] = snap99_id
                    sat_frames.append(df_metric)
                except Exception:
                    pass

            if sat_frames:
                df_satellite_rows = pd.concat(sat_frames, ignore_index=True)
            else:
                df_satellite_rows = pd.DataFrame(columns=["snap99_id", "SnapNum"])

            raw_sat_data = aggregate_satellite_metric_rows(
                df_satellite_rows=df_satellite_rows,
                is_top10=is_top10,
                valid_top10_map=valid_top10_map,
                metrics_keys=metrics_keys,
                metrics_csv_keys=metrics_csv_keys,
                reference_snapshot=99,
            )

            # --- Load Host-Centric rows (evolving population) ---
            host_records = []
            for snap_folder in tqdm(snap_folders, desc="Loading Host-Centric"):
                self.pump_events()
                snap_num = int(snap_folder.split("Snap")[-1])
                host_file_pattern = f"*_snap_{snap_num}_metrics_top_10.csv" if is_top10 else f"*_snap_{snap_num}_metrics.csv"
                host_search_path = os.path.join(snap_folder, "Halo*", host_file_pattern)
                host_metric_files = glob.glob(host_search_path)

                for metric_file in host_metric_files:
                    self.pump_events()
                    try:
                        df_metric = pd.read_csv(metric_file)
                        if df_metric.empty:
                            continue
                        current_id = int(re.search(r'_halo_(\d+)_snap_', metric_file).group(1))
                        row_data = df_metric.iloc[0].to_dict()
                        host_records.append(
                            {
                                "snapshot": snap_num,
                                "current_id": current_id,
                                **row_data,
                            }
                        )
                    except Exception:
                        pass

            if host_records:
                df_host_rows = pd.DataFrame(host_records)
            else:
                df_host_rows = pd.DataFrame(columns=["snapshot", "current_id"])

            # --- Extract Satellite Counts from already loaded Host Data ---
            self.snap99_sat_counts = {}
            if not df_host_rows.empty and 'subhalo_count' in df_host_rows.columns:
                snap99_df = df_host_rows[df_host_rows['snapshot'] == 99]
                self.snap99_sat_counts = dict(zip(snap99_df['current_id'], snap99_df['subhalo_count']))
            # --------------------------------------------------------------

            raw_host_data = aggregate_host_metric_rows(
                df_host_rows=df_host_rows,
                lineage_df=lineage_df,
                is_top10=is_top10,
                valid_top10_map=valid_top10_map,
                metrics_keys=metrics_keys,
                metrics_csv_keys=metrics_csv_keys,
            )

            # --- Execute Aggregation & Processing ---
            self.canvas_systems_table.set_top10_mode(is_top10) 

            sat_processed = process_raw_overview_data(
                raw_data=raw_sat_data,
                metrics_keys=metrics_keys,
                existing_system_table_data=self.system_table_data,
                latest_snapshot=99,
                pairwise_cap=90.0,
            )
            plot_data_sat = sat_processed["plot_data"]
            self.sorted_ids_cache["Sat"] = sat_processed["sorted_ids"]
            self.system_table_data = sat_processed["system_table_data"]

            host_processed = process_raw_overview_data(
                raw_data=raw_host_data,
                metrics_keys=metrics_keys,
                existing_system_table_data=self.system_table_data,
                latest_snapshot=99,
                pairwise_cap=90.0,
            )
            plot_data_host = host_processed["plot_data"]
            self.sorted_ids_cache["Host"] = host_processed["sorted_ids"]
            self.system_table_data = host_processed["system_table_data"]

            # 2. Plotting
            self.canvas_overview_satellite.plot_overview(plot_data_sat, mode="Satellite-Centric (Fixed Progenitors)")
            self.canvas_overview_host.plot_overview(plot_data_host, mode="Host-Centric (Evolving Population)")

            # 3. Update the Top/Bottom Systems Table
            self.update_system_table_tab()

            # --- CACHE DATA AND ENABLE EXPORT ---
            self.cached_overview_data_sat = plot_data_sat
            self.cached_overview_data_host = plot_data_host
            self.export_all_lines_button.setEnabled(True)

            # --- 4. Snapshot 99 Corner Plot — build flat base DataFrame ---
            snap99_metrics = raw_sat_data.get(99, {})

            all_ids_snap99 = set()
            for metric, values in snap99_metrics.items():
                for halo_id, val in values:
                    all_ids_snap99.add(halo_id)

            base_records = []
            for halo_id in all_ids_snap99:
                row_data = {'HaloID': halo_id}
                for metric in metrics_keys:
                    val = np.nan
                    for hid, v in snap99_metrics.get(metric, []):
                        if hid == halo_id:
                            val = v
                            break
                    row_data[metric] = val
                base_records.append(row_data)

            self.snap99_base_df = pd.DataFrame(base_records)

            # --- Populate controls ---
            core_metrics = [m for m in metrics_keys if m != 'Normalised_RMS_Distance']

            self.corner_category_combo.blockSignals(True)
            self.corner_category_combo.clear()
            self.corner_category_combo.addItems(["Majority Vote (>= 3 Metrics)"] + metrics_keys)
            self.corner_category_combo.blockSignals(False)

            self.corner_metric_list.clear()
            for m in core_metrics:
                self.corner_metric_list.addItem(m)

            self.update_corner_plot()

        QMessageBox.information(self, "Plotting Complete", "Metrics Overview distribution plotted.")


    def update_corner_plot(self):
        """Re-categorize and re-render the Snap 99 Corner Plot from the base DataFrame."""
        if not hasattr(self, 'snap99_base_df') or self.snap99_base_df.empty:
            return

        total_halos = len(self.snap99_base_df)
        core_metrics = ["Centroid_Offset", "MRL", "Pairwise", "Hemi_Contrast"]

        # --- Categorization ---
        selected = self.corner_category_combo.currentText()
        top_ids = set()
        bottom_ids = set()

        if selected == "Majority Vote (>= 3 Metrics)" or not self.system_table_data:
            for halo_id in self.snap99_base_df['HaloID']:
                top_count = sum(
                    halo_id in self.system_table_data.get(m, {}).get('Top 1/3', {})
                    for m in core_metrics
                )
                bottom_count = sum(
                    halo_id in self.system_table_data.get(m, {}).get('Bottom 1/3', {})
                    for m in core_metrics
                )
                if top_count >= 3:
                    top_ids.add(halo_id)
                elif bottom_count >= 3:
                    bottom_ids.add(halo_id)
        else:
            metric_data = self.system_table_data.get(selected, {})
            top_ids = set(metric_data.get('Top 1/3', {}).keys())
            bottom_ids = set(metric_data.get('Bottom 1/3', {}).keys())

        # --- Dynamic labels ---
        middle_ids = set(self.snap99_base_df['HaloID']) - top_ids - bottom_ids
        if total_halos > 0:
            self.lbl_top_corner = (
                f"Most Lopsided ({len(top_ids)} | "
                f"{len(top_ids) / total_halos * 100:.1f}%)"
            )
            self.lbl_bot_corner = (
                f"Least Lopsided ({len(bottom_ids)} | "
                f"{len(bottom_ids) / total_halos * 100:.1f}%)"
            )
            self.lbl_mid_corner = (
                f"Remaining Systems ({len(middle_ids)} | "
                f"{len(middle_ids) / total_halos * 100:.1f}%)"
            )
        else:
            self.lbl_top_corner = "Most Lopsided"
            self.lbl_bot_corner = "Least Lopsided"
            self.lbl_mid_corner = "Remaining Systems (0 | 0.0%)"

        # --- Build categorized DataFrame ---
        def assign_category(hid):
            if hid in top_ids:
                return self.lbl_top_corner
            if hid in bottom_ids:
                return self.lbl_bot_corner
            return self.lbl_mid_corner

        df = self.snap99_base_df.copy()
        df['Category'] = df['HaloID'].map(assign_category)
        self.snap99_corner_df = df

        # --- Ordered metrics from list widget ---
        ordered_metrics = [
            self.corner_metric_list.item(i).text()
            for i in range(self.corner_metric_list.count())
        ]

        # --- Optional: transform to empirical percentile ranks ---
        if self.corner_pct_checkbox.isChecked():
            for metric in ordered_metrics:
                if metric in df.columns:
                    df[metric] = df[metric].rank(pct=True)

        # --- Clear canvas area ---
        for i in reversed(range(self.corner_canvas_layout.count())):
            widget = self.corner_canvas_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        if df.empty or not ordered_metrics:
            lbl = QLabel("No data available for the corner plot.")
            lbl.setAlignment(Qt.AlignCenter)
            self.corner_canvas_layout.addWidget(lbl)
            return

        with plt.style.context("dark_background"):
            grid = sns.pairplot(
                df[ordered_metrics + ['Category']],
                vars=ordered_metrics,
                hue='Category',
                palette={
                    self.lbl_top_corner: 'red',
                    self.lbl_bot_corner: 'blue',
                    self.lbl_mid_corner: 'gray',
                },
                corner=True,
                plot_kws={'alpha': 0.6},
            )
            grid.figure.patch.set_facecolor('#1f1f1f')
            for ax in grid.axes.flatten():
                if ax is not None:
                    ax.set_facecolor('#242424')

            canvas = FigureCanvas(grid.figure)
            self.corner_canvas_layout.addWidget(canvas)


    def update_system_table_tab(self):
        """Generates the table of top/bottom systems from self.system_table_data."""
        if not self.system_table_data:
            self.canvas_systems_table.update_table(pd.DataFrame(), {})
            return

        table_df = build_system_table_dataframe(self.system_table_data)
        self.canvas_systems_table.update_table(table_df, getattr(self, 'snap99_sat_counts', {}))


    def load_lineage_summary(self):
        lineage_file = os.path.join(self.base_export_folder, "halo_lineage_summary.csv")
        if not os.path.exists(lineage_file):
            QMessageBox.warning(self, "File Not Found", f"halo_lineage_summary.csv not found in:\n{self.base_export_folder}")
            return
        try:
            # We now require the 'Halo_Category_Snap99' column
            self.lineage_df = pd.read_csv(lineage_file, index_col="HaloID_at_Snap99")
            if 'Halo_Category_Snap99' not in self.lineage_df.columns:
                 QMessageBox.critical(self, "File Error", "The lineage file is missing the 'Halo_Category_Snap99' column. Please run 'interesting_halo_finder.py' first.")
                 return

            self.lineage_combo.blockSignals(True)
            self.lineage_combo.clear()

            # Create a model for the QComboBox to support unselectable items
            model = self.lineage_combo.model()

            # Only one 'Tracked' category is used for display
            category = 'Tracked'
            group_df = self.lineage_df # Now all halos are in one group

            # 1. Add Category Header (Non-selectable)
            header_text = f"--- {category.upper()} ({len(group_df)}) ---"

            header_item = QStandardItem(header_text)
            header_item.setFlags(Qt.NoItemFlags | Qt.ItemIsEnabled)
            model.appendRow(header_item)

            first_selectable_index = -1
            current_index = 1

            # 2. Add System Items
            # Note: The item text still includes the category ('Tracked') for consistency with old formatting
            display_items = [f"{idx} ({row['Halo_Category_Snap99']})"
                             for idx, row in group_df.sort_index().iterrows()]

            for item_text in display_items:
                system_item = QStandardItem(item_text)
                model.appendRow(system_item)
                if first_selectable_index == -1:
                    first_selectable_index = current_index
                current_index += 1

            self.lineage_combo.blockSignals(False)

            # Set the selection to the first selectable item (if any)
            if first_selectable_index != -1:
                self.lineage_combo.setCurrentIndex(first_selectable_index)
                self.on_lineage_selected(first_selectable_index)

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Lineage File", f"Failed to load or parse lineage file:\n{str(e)}")


    def on_lineage_selected(self, index):
        if self.lineage_df.empty or index < 0: return

        selected_text = self.lineage_combo.itemText(index)

        # --- Check if the selected item is a category header ---
        if selected_text.startswith('---'):
            # If it's a header, do nothing (prevent selection/processing)
            return

        # Extract the Halo ID part (before the space)
        selected_id_str = selected_text.split(' ')[0]
        selected_id = int(selected_id_str)
        selected_id_snap99 = selected_id

        lineage_row = self.lineage_df.loc[selected_id]
        self.current_lineage = {}
        for col in self.lineage_df.columns:
            if col.startswith("HaloID_Snap_"):
                halo_id = lineage_row[col]
                if halo_id != -1:
                    snap_num = int(col.split('_')[-1])
                    self.current_lineage[snap_num] = int(halo_id)

        # Determine N_sat at the latest snapshot to gate Top-10 checkboxes
        if self.current_lineage and self.base_export_folder:
            latest_snap = max(self.current_lineage.keys())
            latest_halo_id = self.current_lineage[latest_snap]
            latest_dir = os.path.join(
                self.base_export_folder, f"Snap{latest_snap}", f"Halo {latest_halo_id}"
            )
            if os.path.isdir(latest_dir):
                latest_files = glob.glob(os.path.join(latest_dir, "*.csv"))
                latest_metrics = self._get_metrics_from_files(latest_files, top10=False)
                snap99_n_sat = latest_metrics["N_sat"] if latest_metrics else 0
            else:
                snap99_n_sat = 0
            self._gate_top10_checkboxes(snap99_n_sat)

        # Re-calc lineage metrics (respecting top 10 toggle)
        self.gather_lineage_metrics()
        self.plot_lineage_metrics()

        self.available_snaps_list = []
        is_top10 = self.toggle_top10.isChecked()

        for snap, halo_id in sorted(self.current_lineage.items()):
            # NEW FLAT PATH
            parent_dir = os.path.join(self.base_export_folder, f"Snap{snap}", f"Halo {halo_id}")
            # Check if the halo's folder exists (a good proxy for data existence)
            if os.path.isdir(parent_dir):
                # If Top 10 Only is on, verify the top 10 file exists before adding snap
                if is_top10:
                    top10_file = glob.glob(os.path.join(parent_dir, f"*_snap_{snap}_*top_10.csv"))
                    if top10_file:
                        self.available_snaps_list.append(snap)
                else:
                    self.available_snaps_list.append(snap)
            # End of new logic

        sorted_snaps = self.available_snaps_list

        if not sorted_snaps:
            QMessageBox.warning(self, "No Data Available", "No exported data found for this halo lineage (matching current filters).")
            # Reset the slider to a neutral state if no data is available
            self.snapshot_slider.blockSignals(True)
            self.snapshot_slider.setMinimum(0)
            self.snapshot_slider.setMaximum(0)
            self.snapshot_slider.setValue(0)
            self.snapshot_slider.blockSignals(False)
            self.snapshot_label.setText("N/A")
            self.canvas_3d.clear(); self.canvas_3d.draw()
            return

        # Map the sparse snapshot numbers to continuous slider indices (0 to len-1)
        self.snapshot_slider.blockSignals(True)
        self.snapshot_slider.setMinimum(0)
        self.snapshot_slider.setMaximum(len(sorted_snaps) - 1)

        # Set the value to the index of the latest available snap
        latest_snap_index = len(sorted_snaps) - 1
        self.snapshot_slider.setValue(latest_snap_index)
        self.snapshot_slider.blockSignals(False)

        # Pass the index (latest_snap_index) to the update function
        self.update_3d_view_for_lineage_snap(latest_snap_index)

        # Lazy evaluation: mark p-value tab dirty and only calculate if currently visible
        self._p_value_dirty = True
        if self.plot_tabs.currentWidget() == self.p_value_tab:
            self.update_p_value_tab_tracked()
            self._p_value_dirty = False

    
    def on_top10_toggled(self):
        # Refresh current view based on active mode
        if self.mode_tracked_history.isChecked():
            # Re-trigger lineage selection logic to update available snaps list
            self.on_lineage_selected(self.lineage_combo.currentIndex())
        elif self.mode_single_halo.isChecked():
            self.update_current_view()


    def plot_lineage_metrics(self):
        # Plotting the Host-Centric evolution (individual halo metrics)
        self.canvas_metrics.plot_evolution(self.lineage_metrics)
        self.plot_tabs.setTabText(1, "Lineage Metric Evolution")


    def focus_on_track(self, track_id):
        self.canvas_3d.show_single_track(track_id)
        self.focus_reset_button.show()


    def back_to_default(self):
        self.update_current_view()
        self.focus_reset_button.hide()
        self.canvas_3d.focused_track_id = None
        self.subhalo_info_label.hide()
        self.subhalo_info_text.hide()
        self.canvas_3d.dark_info_text = None


    def update_subhalo_info(self, info_text):
        self.subhalo_info_label.show()
        self.subhalo_info_text.show()
        self.subhalo_info_text.setText(info_text)


    def update_3d_view_for_lineage_snap(self, index):
        if not hasattr(self, 'available_snaps_list') or not self.available_snaps_list:
             return

        if not (0 <= index < len(self.available_snaps_list)):
             # Safety check in case of unexpected index value
             return

        # Map the slider index back to the actual snap number
        snap_num = self.available_snaps_list[index]
        halo_id = self.current_lineage[snap_num] # Use the actual snap_num to look up the halo_id

        self.snapshot_label.setText(f"Snapshot: {snap_num} (Halo ID: {halo_id})")

        # Determine the Category from the Snap99 data for file lookup
        selected_id_str = self.lineage_combo.currentText().split(' ')[0]
        if not selected_id_str or selected_id_str.startswith('---'):
            return

        # The files are now located in SnapXX/Halo ID/
        parent_dir = os.path.join(self.base_export_folder, f"Snap{snap_num}", f"Halo {halo_id}") # NEW FLAT PATH

        files_to_plot = []
        metrics_file_path = None

        # Check Top 10 Toggle
        is_top10 = self.toggle_top10.isChecked()

        # The folder is guaranteed to exist due to the filter in on_lineage_selected
        if os.path.isdir(parent_dir):
            if is_top10:
                # Top 10 specific files - WILDCARD SEARCH to support any sim prefix
                files_to_plot = glob.glob(os.path.join(parent_dir, f"*_snap_{snap_num}_*top_10.csv"))

                # Metrics file
                metric_candidates = glob.glob(os.path.join(parent_dir, f"*_snap_{snap_num}_metrics_top_10.csv"))
                if metric_candidates:
                    metrics_file_path = metric_candidates[0]
                else:
                    standard_metric_candidates = glob.glob(os.path.join(parent_dir, f"*_snap_{snap_num}_metrics.csv"))
                    if standard_metric_candidates:
                        metrics_file_path = standard_metric_candidates[0]

                # We need central IDs too, which are standard
                central_candidates = glob.glob(os.path.join(parent_dir, f"*_snap_{snap_num}_central_ids.csv"))
                if central_candidates: files_to_plot.append(central_candidates[0])
            else:
                # Standard files
                files_to_plot = glob.glob(os.path.join(parent_dir, f"*_snap_{snap_num}_*.csv"))
                # Filter out top 10 if present in standard view
                files_to_plot = [f for f in files_to_plot if "top_10" not in f]

                metric_candidates = glob.glob(os.path.join(parent_dir, f"*_snap_{snap_num}_metrics.csv"))
                if metric_candidates: metrics_file_path = metric_candidates[0]

            if metrics_file_path and os.path.exists(metrics_file_path):
                files_to_plot.append(metrics_file_path)

        # Read and display single-snap metrics before plotting 3D view
        central_mass, group_r_crit200 = self.read_single_snap_metrics(metrics_file_path)

        # Now pass the files for 3D plotting
        self.plot_from_files(files_to_plot, central_mass, group_r_crit200=group_r_crit200)


    def update_current_view(self, _=None):
        if self.mode_single_halo.isChecked():
            # In Single Halo mode, we rely on the specific folder selected in self.current_halo_folder
            if not hasattr(self, 'current_halo_folder') or not self.current_halo_folder: return

            items = [self.file_list.item(i).text() for i in range(self.file_list.count())]

            if items:
                # The directory is already specific (e.g., .../Snap99/Halo 488)
                base_dir = self.current_halo_folder

                # Check Top 10 Toggle
                is_top10 = self.toggle_top10.isChecked()

                if is_top10:
                    # Filter for Top 10 files
                    files_to_plot = [os.path.join(base_dir, f) for f in items if "top_10" in f or "central_ids" in f]
                    # If we don't find them in the list (maybe list wasn't refreshed), check dir directly
                    if len(files_to_plot) <= 1: # just central_ids
                         files_to_plot = glob.glob(os.path.join(base_dir, "*top_10.csv")) + glob.glob(os.path.join(base_dir, "*central_ids.csv"))
                else:
                    # Standard files (exclude top 10)
                    files_to_plot = [os.path.join(base_dir, f) for f in items if "top_10" not in f]

                # --- Extract ID/Snap to find metrics/evolution files by name ---
                # We need to find a valid file to parse name from
                valid_file = next((f for f in files_to_plot if "positions" in f and "dark" not in f), None)
                if not valid_file:
                    valid_file = next((f for f in files_to_plot if "positions" in f), None)

                if valid_file:
                    base_name = os.path.basename(valid_file)
                    # Regex to handle both _positions.csv and _positions_top_10.csv
                    match = re.match(r'(.*_snap_\d+)_positions.*\.csv', base_name)

                    if match:
                        file_prefix = match.group(1) # e.g., sim_TNG100-1_halo_488_snap_99

                        # Construct and read single-snapshot metrics file path
                        if is_top10:
                            metrics_file_name = f"{file_prefix}_metrics_top_10.csv"
                            evol_file_name = f"{file_prefix}_metric_evolution_top_10.csv"
                        else:
                            metrics_file_name = f"{file_prefix}_metrics.csv"
                            evol_file_name = f"{file_prefix}_metric_evolution.csv"

                        metrics_file_path = os.path.join(base_dir, metrics_file_name)

                        # Read and display single-snap metrics, getting 3D context
                        central_mass, group_r_crit200 = self.read_single_snap_metrics(metrics_file_path)

                        # Construct and add metric evolution file path to files_to_plot
                        evol_file_path = os.path.join(base_dir, evol_file_name)

                        if os.path.exists(evol_file_path):
                            files_to_plot.append(evol_file_path)

                        # Add metrics file itself to list for plot_from_files (to ignore it safely but ensure passed)
                        if os.path.exists(metrics_file_path):
                            files_to_plot.append(metrics_file_path)

                        # Plot the 3D view and update Metric History
                        self.plot_from_files(files_to_plot, central_mass, group_r_crit200=group_r_crit200)
                    else:
                        cand = glob.glob(os.path.join(base_dir, "*_metrics.csv"))
                        if cand:
                            c_mass, r_crit = self.read_single_snap_metrics(cand[0])
                            self.plot_from_files(files_to_plot, c_mass, group_r_crit200=r_crit)
                        else:
                            self.plot_from_files(files_to_plot)
                else:
                    cand = glob.glob(os.path.join(base_dir, "*_metrics.csv"))
                    if cand:
                        c_mass, r_crit = self.read_single_snap_metrics(cand[0])
                        self.plot_from_files(files_to_plot, c_mass, group_r_crit200=r_crit)
                    else:
                        self.plot_from_files(files_to_plot) # Plot without metrics/evolution

                # Update Statistical Significance tab with all files in this folder
                all_files_in_dir = [os.path.join(base_dir, f)
                                    for f in [self.file_list.item(i).text()
                                              for i in range(self.file_list.count())]]
                # Cache files and directory for lazy evaluation & recalculate handlers
                self._cached_p_val_files = all_files_in_dir
                self._current_halo_dir = base_dir
                self._p_value_dirty = True
                self._sat_centric_dirty = True

                # Gate top 10 checkboxes immediately based on luminous satellite count
                metrics_snap = self._get_metrics_from_files(all_files_in_dir, top10=False)
                if metrics_snap:
                    self._gate_top10_checkboxes(metrics_snap["N_sat"])

                # Lazy evaluation: only compute if the tab is currently active
                if self.plot_tabs.currentWidget() == self.p_value_tab:
                    self.update_p_value_tab_single(all_files_in_dir)
                    self._p_value_dirty = False
                if self.plot_tabs.currentWidget() == self.sat_centric_tab:
                    self.update_sat_centric_tab(all_files_in_dir)
                    self._sat_centric_dirty = False
            else:
                self.clear_plots(clear_metrics_history=False)

        elif self.mode_tracked_history.isChecked():
            self.update_3d_view_for_lineage_snap(self.snapshot_slider.value())


    def plot_from_files(self, files, central_mass=50.0, group_r_crit200=0.0):
        if not files: return
        self.clear_plots(clear_metrics_history=False, clear_lopsided=False)

        metric_evolution_data = None
        files_for_3d_plot = []

        for file in files:
            name = os.path.basename(file)

            # Load Metric Evolution data (This contains the data for the Satellite-Centric plot)
            if "metric_evolution" in name: # Matches _metric_evolution.csv and _metric_evolution_top_10.csv
                try:
                    df_evol = pd.read_csv(file)
                    
                    base_dir = self.base_export_folder
                    if not base_dir:
                        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(file)))
                    
                    snap_folders = glob.glob(os.path.join(base_dir, "Snap*"))
                    if snap_folders:
                        valid_snaps = [int(os.path.basename(f).replace("Snap", "")) for f in snap_folders]
                        df_evol = df_evol[df_evol['SnapNum'].isin(valid_snaps)]
                        
                    metric_evolution_data = df_evol.to_dict(orient='list')
                except Exception as e:
                    print(f"Error loading metric evolution file {file}: {e}")
            else:
                # All other files (positions, dark positions, tracks, central_ids, metrics) go to 3D plot
                files_for_3d_plot.append(file)


        # 1. Update 3D plot
        self.canvas_3d.plot_from_files(files_for_3d_plot, central_mass,
                                       show_dark=self.toggle_dark.isChecked(),
                                       show_tracks=self.toggle_tracks.isChecked(),
                                       group_r_crit200=group_r_crit200)

        # 2. Update Metric Evolution Plot based on mode
        if self.mode_tracked_history.isChecked():
            # Lineage mode: use the pre-loaded self.lineage_metrics (Host-Centric)
            self.plot_lineage_metrics()
        elif self.mode_single_halo.isChecked():
            # Single Halo mode: use the data loaded from the current directory (Satellite-Centric)
            if metric_evolution_data:
                self.canvas_metrics.plot_evolution(metric_evolution_data, title_suffix="(Satellite-Centric)")
            else:
                self.canvas_metrics.clear_and_show_message("Metric Evolution file not found in current directory.")


    def export_all_lines_plot(self):
        if not hasattr(self, 'cached_overview_data_sat'):
            QMessageBox.warning(self, "No Data", "Please load the Metrics Overview data first.")
            return

        warning = QMessageBox()
        warning.setIcon(QMessageBox.Warning)
        warning.setWindowTitle("Confirm Computational Irresponsibility")
        warning.setText("Plotting all individual tracks for all systems (Host-Centric and Satellite-Centric) is computationally intensive.")
        warning.setInformativeText("Do you wish to continue?")
        warning.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        warning.setDefaultButton(QMessageBox.No)

        if warning.exec_() == QMessageBox.No:
            return

        export_dir = os.path.join(self.base_export_folder, "metrics_overview_plots")
        os.makedirs(export_dir, exist_ok=True)

        modes = [
            ("Satellite-Centric (Fixed Progenitors)", self.cached_overview_data_sat, 'Sat'),
            ("Host-Centric (Evolving Population)", self.cached_overview_data_host, 'Host')
        ]

        table_data = self.system_table_data

        # Naming suffix
        tag = "_top_10" if self.toggle_top10.isChecked() else ""

        all_snaps = []
        for _, aggregated_data, _ in modes:
            for category, data in aggregated_data.items():
                if data.get("SnapNum"):
                    all_snaps.extend(data['SnapNum'])
        
        min_snap = min(all_snaps) if all_snaps else None
        max_snap = max(all_snaps) if all_snaps else None

        metrics_list = [
            ('Centroid_Offset', 'Centroid Offset', 0),
            ('MRL', 'MRL', 1),
            ('Pairwise', 'Pairwise Angle', 2),
            ('Hemi_Contrast', 'Hemi Contrast', 3),
            ('Normalised_RMS_Distance', 'Norm. RMS Distance', 4),
        ]

        def draw_metric_on_axis(ax, data_category_dict, cache_key, metric_key, ylabel):
            if not data_category_dict.get("SnapNum") or metric_key not in data_category_dict or not data_category_dict[metric_key]:
                return
                
            snap_nums = data_category_dict['SnapNum']
            master_ids = self.sorted_ids_cache.get(cache_key, [])
            
            most_lopsided_ids = set()
            least_lopsided_ids = set()
            if metric_key in table_data:
                most_lopsided_ids = set(table_data[metric_key]['Top 1/3'].keys())
                least_lopsided_ids = set(table_data[metric_key]['Bottom 1/3'].keys())

            metric_array = np.array(data_category_dict[metric_key])

            if metric_array.size > 0:
                tracks_data = metric_array.T
                
                for track_idx, track_values in enumerate(tracks_data):
                    if track_idx >= len(master_ids): break
                    current_id = master_ids[track_idx]

                    plot_color = CATEGORY_COLORS['Tracked']
                    plot_alpha = 0.05

                    if current_id in most_lopsided_ids:
                        plot_color = CATEGORY_COLORS['Most Lopsided (Top 1/3)']
                        plot_alpha = 0.15
                    elif current_id in least_lopsided_ids:
                        plot_color = CATEGORY_COLORS['Least Lopsided (Bottom 1/3)']
                        plot_alpha = 0.08

                    if np.sum(~np.isnan(track_values)) > 1:
                        ax.plot(snap_nums, track_values, color=plot_color, linewidth=0.5, alpha=plot_alpha, zorder=0)

                stats = compute_percentile_band_from_matrix(
                    metric_array,
                    lower_percentile=16,
                    upper_percentile=84,
                    axis=1,
                )
                median = stats["median"]
                low_bound = stats["low_bound"]
                high_bound = stats["high_bound"]

                ax.plot(snap_nums, median, linestyle='-', color=CATEGORY_COLORS['Tracked'], linewidth=2.5, label='Tracked Median', zorder=1)
                ax.fill_between(snap_nums, low_bound, high_bound, color=CATEGORY_COLORS['Tracked'], alpha=0.4, linewidth=0, zorder=1)

                top_indices = [idx for idx, hid in enumerate(master_ids) if hid in most_lopsided_ids and idx < metric_array.shape[1]]
                bottom_indices = [idx for idx, hid in enumerate(master_ids) if hid in least_lopsided_ids and idx < metric_array.shape[1]]

                if top_indices:
                    top_stats = compute_percentile_band_from_matrix(
                        metric_array[:, top_indices],
                        lower_percentile=16,
                        upper_percentile=84,
                        axis=1,
                    )
                    ax.plot(snap_nums, top_stats["median"], linestyle='--', linewidth=2.5, color=CATEGORY_COLORS['Most Lopsided (Top 1/3)'], zorder=2)
                
                if bottom_indices:
                    bottom_stats = compute_percentile_band_from_matrix(
                        metric_array[:, bottom_indices],
                        lower_percentile=16,
                        upper_percentile=84,
                        axis=1,
                    )
                    ax.plot(snap_nums, bottom_stats["median"], linestyle='--', linewidth=2.5, color=CATEGORY_COLORS['Least Lopsided (Bottom 1/3)'], zorder=2)

            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', rotation=45)
            if min_snap is not None and max_snap is not None:
                ax.set_xlim(min_snap, max_snap)

        legend_handles = [
            Line2D([0], [0], color=CATEGORY_COLORS['Tracked'], linewidth=2.5, label='Tracked Median'),
            Line2D([0], [0], color=CATEGORY_COLORS['Most Lopsided (Top 1/3)'], linestyle='--', linewidth=2.5, label='Top 1/3 Median'),
            Line2D([0], [0], color=CATEGORY_COLORS['Least Lopsided (Bottom 1/3)'], linestyle='--', linewidth=2.5, label='Bottom 1/3 Median'),
            Line2D([0], [0], color=CATEGORY_COLORS['Most Lopsided (Top 1/3)'], linewidth=0.5, alpha=1.0, label='Most Lopsided (Top 1/3)'),
            Line2D([0], [0], color=CATEGORY_COLORS['Least Lopsided (Bottom 1/3)'], linewidth=0.5, alpha=1.0, label='Least Lopsided (Bottom 1/3)'),
            Line2D([0], [0], color=CATEGORY_COLORS['Tracked'], linewidth=0.5, alpha=1.0, label='Remaining Systems')
        ]

        main_metrics = metrics_list[:4]
        all_metrics = metrics_list

        has_corner = (
            hasattr(self, 'snap99_corner_df')
            and not self.snap99_corner_df.empty
            and hasattr(self, 'corner_metric_list')
            and self.corner_metric_list.count() > 0
        )
        total_plots = len(modes) * 2 + len(all_metrics) + (1 if has_corner else 0)

        with self.busy_state("Rendering and Exporting Overview Figures..."):
            with tqdm(
                total=total_plots,
                desc="[Export] Generating overview plots",
                unit="plot",
                leave=False,
            ) as export_pbar:
                for mode_title, aggregated_data, cache_key in modes:
                    data = aggregated_data.get('Tracked', {})

                    # --- 2x2 Layout ---
                    fig_2x2, axes_2x2 = plt.subplots(2, 2, figsize=(16, 12))
                    axes_flat_2x2 = axes_2x2.flatten()
                    for i, (metric_key, label, _) in enumerate(main_metrics):
                        draw_metric_on_axis(axes_flat_2x2[i], data, cache_key, metric_key, label)
                    
                    axes_flat_2x2[0].legend(handles=legend_handles, fontsize='small')
                    fig_2x2.suptitle(f'Metrics Overview (2x2): {mode_title} - All Individual Tracks', fontsize=16)
                    fig_2x2.supxlabel('Snapshot Number')
                    fig_2x2.tight_layout(pad=1.0, h_pad=0.5, rect=[0, 0, 1, 0.96])
                    
                    filename_2x2 = f"{cache_key}_2x2_Plot{tag}.png"
                    fig_2x2.savefig(os.path.join(export_dir, filename_2x2), dpi=300)
                    plt.close(fig_2x2)
                    export_pbar.update(1)
                    self.pump_events()

                    # --- 4x1 Layout ---
                    fig_4x1, axes_4x1 = plt.subplots(4, 1, figsize=(10, 18), sharex=True)
                    for i, (metric_key, label, _) in enumerate(main_metrics):
                        draw_metric_on_axis(axes_4x1[i], data, cache_key, metric_key, label)
                    
                    axes_4x1[0].legend(handles=legend_handles, fontsize='small')
                    fig_4x1.suptitle(f'Metrics Overview (4x1): {mode_title} - All Individual Tracks', fontsize=16)
                    fig_4x1.supxlabel('Snapshot Number')
                    fig_4x1.tight_layout(pad=1.0, h_pad=0.5, rect=[0, 0, 1, 0.96])

                    filename_4x1 = f"{cache_key}_4x1_Plot{tag}.png"
                    fig_4x1.savefig(os.path.join(export_dir, filename_4x1), dpi=300)
                    plt.close(fig_4x1)
                    export_pbar.update(1)
                    self.pump_events()


                # --- 1x2 Side-by-Side Layout ---
                sat_data = modes[0][1].get('Tracked', {})
                host_data = modes[1][1].get('Tracked', {})

                for metric_key, label, _ in all_metrics:
                    fig_comp, axes_comp = plt.subplots(1, 2, figsize=(18, 6))

                    # Satellite
                    draw_metric_on_axis(axes_comp[0], sat_data, 'Sat', metric_key, label)
                    axes_comp[0].set_title('Satellite-Centric (Fixed Progenitors)')

                    # Host
                    draw_metric_on_axis(axes_comp[1], host_data, 'Host', metric_key, label)
                    axes_comp[1].set_title('Host-Centric (Evolving Population)')

                    if metric_key == 'Normalised_RMS_Distance':
                        current_handles = [
                            Line2D([0], [0], color=CATEGORY_COLORS['Tracked'], linewidth=2.5, label='Tracked Median'),
                            Line2D([0], [0], color=CATEGORY_COLORS['Most Lopsided (Top 1/3)'], linestyle='--', linewidth=2.5, label='Top 1/3 Median'),
                            Line2D([0], [0], color=CATEGORY_COLORS['Least Lopsided (Bottom 1/3)'], linestyle='--', linewidth=2.5, label='Bottom 1/3 Median'),
                            Line2D([0], [0], color=CATEGORY_COLORS['Most Lopsided (Top 1/3)'], linewidth=0.5, alpha=1.0, label='Most bound (Top 1/3)'),
                            Line2D([0], [0], color=CATEGORY_COLORS['Least Lopsided (Bottom 1/3)'], linewidth=0.5, alpha=1.0, label='Least bound (Bottom 1/3)'),
                            Line2D([0], [0], color=CATEGORY_COLORS['Tracked'], linewidth=0.5, alpha=1.0, label='Remaining Systems')
                        ]
                    else:
                        current_handles = legend_handles

                    axes_comp[0].legend(handles=current_handles, fontsize='small')
                    fig_comp.suptitle(f'{label} Comparison: Satellite vs Host', fontsize=16)
                    fig_comp.supxlabel('Snapshot Number')
                    fig_comp.tight_layout(pad=1.0, h_pad=0.5, rect=[0, 0, 1, 0.96])

                    filename_comp = f"Comparison_{metric_key}{tag}.png"
                    fig_comp.savefig(os.path.join(export_dir, filename_comp), dpi=300)
                    plt.close(fig_comp)
                    export_pbar.update(1)
                    self.pump_events()

                if has_corner:
                    ordered_metrics = [
                        self.corner_metric_list.item(i).text()
                        for i in range(self.corner_metric_list.count())
                    ]
                    if ordered_metrics:
                        export_df = self.snap99_corner_df[ordered_metrics + ['Category']]
                        if self.corner_pct_checkbox.isChecked():
                            export_df = export_df.copy()
                            for metric in ordered_metrics:
                                if metric in export_df.columns:
                                    export_df[metric] = export_df[metric].rank(pct=True)
                        grid = sns.pairplot(
                            export_df,
                            vars=ordered_metrics,
                            hue='Category',
                            palette={
                                self.lbl_top_corner: 'red',
                                self.lbl_bot_corner: 'blue',
                                self.lbl_mid_corner: 'gray',
                            },
                            corner=True,
                            plot_kws={'alpha': 0.6},
                        )
                        grid.figure.suptitle("Snapshot 99 Metric Covariances", y=1.02, fontsize=16)
                        filename_corner = f"Corner_Plot_Snap99{tag}.png"
                        grid.figure.savefig(os.path.join(export_dir, filename_corner), dpi=300, bbox_inches='tight')
                        plt.close(grid.figure)
                    export_pbar.update(1)
                    self.pump_events()

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Export Complete")
        msg_box.setText("All plots generated and exported successfully.")
        msg_box.setInformativeText(export_dir)
        open_button = msg_box.addButton("Open Folder", QMessageBox.ActionRole)
        msg_box.addButton(QMessageBox.Ok)
        
        msg_box.exec_()
        
        if msg_box.clickedButton() == open_button:
            os.startfile(export_dir)

        self.export_all_lines_button.setEnabled(False)


    def clear_plots(self, clear_metrics_history=True, clear_lopsided=True):
        self.canvas_3d.ax.clear(); self.canvas_3d.ax.set_title("Cleared"); self.canvas_3d.draw()
        if clear_metrics_history:
            self.canvas_metrics.clear_and_show_message("Cleared")
        if clear_lopsided:
            self.canvas_overview_satellite.clear_and_show_message("Cleared")
            self.canvas_overview_host.clear_and_show_message("Cleared")
            self.canvas_systems_table.update_table(pd.DataFrame(), {}) # Clear the table
        # Always reset the p-value tab when clearing plots
        if clear_metrics_history or clear_lopsided:
            if hasattr(self, 'p_value_tab'):
                self.p_value_tab.canvas.clear_and_show_message("Load a halo to compute p-values.")
                self.p_value_tab.lbl_status.setText("No data loaded")
                self.p_value_tab.lbl_status.setStyleSheet("color: #888888; font-style: italic; font-size: 11px;")
                self.p_value_tab.table.setVisible(True)
                self.p_value_tab._clear_table()
            # Also reset sat-centric tab
            if hasattr(self, 'sat_centric_tab'):
                self.sat_centric_tab.show_message("Load a halo to see satellite-centric significance.")
            self._current_halo_dir = None
            self._p_value_dirty = True
            self._sat_centric_dirty = True
            self._cached_p_val_files = []


    def clear_all(self):
        self.file_list.clear()
        self.lineage_combo.clear()
        self.snapshot_slider.setValue(0)
        self.snapshot_label.setText("N/A")
        self.metrics_text.clear()
        self.subhalo_info_text.clear()
        self.subhalo_info_label.hide()
        self.subhalo_info_text.hide()
        self.current_lineage = {}
        self.lineage_metrics = {}
        self.base_export_folder = None
        if hasattr(self, 'current_halo_folder'): del self.current_halo_folder
        # Re-init engine to fallback cwd cache after folder is cleared
        self._init_p_value_engine()
        self._p_value_dirty = True
        self._sat_centric_dirty = True
        self._cached_p_val_files = []
        self.clear_plots(clear_metrics_history=True, clear_lopsided=True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HaloReaderApp()
    window.show()
    sys.exit(app.exec_())
