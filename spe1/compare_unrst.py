
import argparse
from pathlib import Path
from ecl.eclfile import EclFile
from ecl.grid import EclGrid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import os
import re

def export_unrst_to_csv(unrst_file: str, output_csv: str, selected_dates=None):
    """
    Extract SWAT and OILKR data from an Eclipse UNRST file and export to CSV.
    Filters timesteps based on selected_dates if provided.
    """
    unrst_path = Path(unrst_file)
    out_csv = Path(output_csv)

    # Load restart file
    unrst = EclFile(str(unrst_path))
    dates = unrst.dates
    available_keys = set(unrst.keys())
    print(f"[INFO] Available keywords in {unrst_file}: {sorted(available_keys)}")

    # Locate grid (EGRID preferred)
    stem = unrst_path.with_suffix("")
    grid_path = None
    for ext in (".EGRID", ".GRID"):
        cand = stem.with_suffix(ext)
        if cand.exists():
            grid_path = cand
            break
    if grid_path is None:
        raise FileNotFoundError(f"Couldn't find {stem.name}.EGRID or {stem.name}.GRID next to the UNRST file.")
    grid = EclGrid(str(grid_path))

    # Validate SWAT keyword
    if "SWAT" not in available_keys or len(unrst["SWAT"]) == 0:
        raise KeyError("SWAT keyword is missing or empty in the UNRST file.")

    nactive = len(unrst["SWAT"][0])
    ijk = np.empty((nactive, 3), dtype=np.int32)
    for ai in range(nactive):
        i0, j0, k0 = grid.get_ijk(active_index=ai)
        ijk[ai] = (i0 + 1, j0 + 1, k0 + 1)

    # Helper for safe keyword fetch
    def get_kw_step(kw_name, step_idx, n_target):
        if kw_name not in available_keys:
            return np.zeros(n_target, dtype=float)
        kw_all = unrst[kw_name]
        if step_idx >= len(kw_all):
            return np.zeros(n_target, dtype=float)
        arr = np.asarray(kw_all[step_idx], dtype=float)
        if arr.size == n_target:
            return arr
        elif arr.size > n_target:
            return arr[:n_target]
        else:
            out = np.zeros(n_target, dtype=float)
            out[:arr.size] = arr
            return out

    # Build dataframe
    frames = []
    for step_idx, date in enumerate(dates):
        date_str = date.strftime("%d.%b %Y")
        if selected_dates and date_str not in selected_dates:
            continue
        if step_idx >= len(unrst["SWAT"]):
            continue
        swat = np.asarray(unrst["SWAT"][step_idx], dtype=float)
        n = swat.size
        ijk_step = ijk[:n]
        oilkr = get_kw_step("OILKR", step_idx, n)
        df_step = pd.DataFrame({
            "I": ijk_step[:, 0],
            "J": ijk_step[:, 1],
            "K": ijk_step[:, 2],
            "SWAT": np.round(swat, 5),
            "OILKR": np.round(oilkr, 5),
            "Group Index": step_idx + 1,
            "Group Description": date_str,
        })
        frames.append(df_step)

    if not frames:
        raise ValueError("No data found for the specified dates.")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Exported {len(df)} rows to {out_csv}")

    # --- Add Turning Point Detection and Process Columns ---
    df_sorted = df.sort_values(by=['I', 'J', 'K', 'Group Index']).reset_index(drop=True)
    turning_points = []
    turning_values = []
    filled_values = []
    process = []
    last_turning_value = df_sorted.loc[0, 'SWAT']
    prev_diff = 0
    for i in range(len(df_sorted)):
        if i == 0:
            turning_points.append('')
            turning_values.append('')
            filled_values.append(last_turning_value)
            process.append('Initial')
            continue
        diff = df_sorted.loc[i, 'SWAT'] - df_sorted.loc[i - 1, 'SWAT']
        if prev_diff != 0 and diff * prev_diff < 0:
            turning_points.append('Turning Point')
            turning_values.append(df_sorted.loc[i, 'SWAT'])
            last_turning_value = df_sorted.loc[i, 'SWAT']
        else:
            turning_points.append('')
            turning_values.append('')
        filled_values.append(last_turning_value)
        if diff > 0:
            process.append('Imbibition')
        elif diff < 0:
            process.append('Drainage')
        else:
            process.append(process[-1])
        prev_diff = diff

    df_sorted['Turning Point'] = turning_points
    df_sorted['Turning Point Value'] = turning_values
    df_sorted['Turning Point Value Filled'] = filled_values
    df_sorted['Process'] = process

    df_sorted.to_csv(out_csv, index=False)
    print(f"[INFO] Added turning point and process columns. Saved to {out_csv}")
    return df_sorted


def compare_csv_plots(files, legends, output_dir="comparison_plots", cells=None):
    os.makedirs(output_dir, exist_ok=True)
    dfs = [pd.read_csv(f).sort_values(by=['Group Index']) for f in files]

    all_dates = sorted(set().union(*[df['Group Description'].unique() for df in dfs]))
    cmap = cm.get_cmap('tab20', len(all_dates))
    date_color_map = {d: cmap(i) for i, d in enumerate(all_dates)}

    if cells is None:
        all_cells = set()
        for df in dfs:
            all_cells.update(df['I'].unique())
        cells = sorted(all_cells)[:10]

    markers = ['o', 'x', '^', 's']
    for cell_i in cells:
        plt.figure(figsize=(12, 7))
        for idx, df in enumerate(dfs):
            cell_data = df[(df['I'] == cell_i) & (df['J'] == 1) & (df['K'] == 1)]
            if cell_data.empty:
                continue
            plt.plot(cell_data['SWAT'], cell_data['OILKR'], linewidth=2, label=legends[idx])
            for _, row in cell_data.iterrows():
                plt.scatter(row['SWAT'], row['OILKR'],
                            color=date_color_map[row['Group Description']],
                            s=80, marker=markers[idx])
        plt.title(f"SWAT vs OILKR for Cell I={cell_i}, J=1, K=1")
        plt.xlabel("SWAT")
        plt.ylabel("OILKR")
        file_legend = plt.legend(loc='upper left', title="Models")
        legend_patches = [mpatches.Patch(color=color, label=date) for date, color in date_color_map.items()]
        plt.gca().add_artist(file_legend)
        plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title="Dates")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cell_I{cell_i}_J1_K1.png", dpi=300, bbox_inches='tight')
        plt.close()
    print(f"[INFO] Plots saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SWAT vs OILKR from up to 4 UNRST files.")
    parser.add_argument("file1", help="First UNRST file")
    parser.add_argument("file2", help="Second UNRST file")
    parser.add_argument("file3", nargs="?", default=None, help="Optional third UNRST file")
    parser.add_argument("file4", nargs="?", default=None, help="Optional fourth UNRST file")
    parser.add_argument("--legends", nargs="+", help="Custom legend labels for files", required=False)
    parser.add_argument("--cells", nargs="+", type=int, help="List of I indices to plot")
    parser.add_argument("--outdir", default="comparison_plots", help="Output directory for plots")
    parser.add_argument("--dates", nargs="+", help="List of dates to extract (format: DD.MMM YYYY)")
    parser.add_argument("--list-dates", action="store_true", help="List all available dates and exit")
    args = parser.parse_args()

    # Handle --list-dates
    if args.list_dates:
        unrst_files = [args.file1, args.file2, args.file3, args.file4]
        for f in unrst_files:
            if f:
                unrst = EclFile(str(Path(f)))
                print(f"\n[INFO] Available dates in {f}:")
                for d in unrst.dates:
                    print("  ", d.strftime("%d.%b %Y"))
        exit(0)

    # Flexible date validation
    if args.dates:
        normalized_dates = []
        for d in args.dates:
            d_clean = d.strip().replace('"', '').replace("'", '')
            parts = d_clean.split()
            if len(parts) == 2:
                day_month = parts[0].split('.')
                if len(day_month) == 2:
                    day = day_month[0]
                    month = day_month[1].capitalize()
                    year = parts[1]
                    d_clean = f"{day}.{month} {year}"
            if re.match(r"^\d{2}\.[A-Za-z]{3} \d{4}$", d_clean):
                normalized_dates.append(d_clean)
            else:
                print(f"[WARNING] Invalid date format after normalization: {d}. Expected DD.MMM YYYY")
        if not normalized_dates:
            raise ValueError("No valid dates provided after normalization.")
        args.dates = normalized_dates

    csv_files = []
    unrst_files = [args.file1, args.file2, args.file3, args.file4]
    for i, f in enumerate(unrst_files):
        if f:
            csv_name = f"swat_oilkr{i+1}.csv"
            export_unrst_to_csv(f, csv_name, selected_dates=args.dates)
            csv_files.append(csv_name)

    if args.legends:
        legends = args.legends
        if len(legends) != len(csv_files):
            raise ValueError("Number of legends must match number of files.")
    else:
        legends = [f"File {i+1}" for i in range(len(csv_files))]

    cells = args.cells if args.cells else None
    compare_csv_plots(csv_files, legends, output_dir=args.outdir, cells=cells)




# import argparse
# from pathlib import Path
# from ecl.eclfile import EclFile
# from ecl.grid import EclGrid
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.patches as mpatches
# import os


# def export_unrst_to_csv(unrst_file: str, output_csv: str):
#     """
#     Extract SWAT and OILKR data from an Eclipse UNRST file and export to CSV.
#     """
#     unrst_path = Path(unrst_file)
#     out_csv = Path(output_csv)

#     # Load restart file
#     unrst = EclFile(str(unrst_path))
#     dates = unrst.dates
#     available_keys = set(unrst.keys())
#     print(f"[INFO] Available keywords in {unrst_file}: {sorted(available_keys)}")

#     # Locate grid (EGRID preferred)
#     stem = unrst_path.with_suffix("")
#     grid_path = None
#     for ext in (".EGRID", ".GRID"):
#         cand = stem.with_suffix(ext)
#         if cand.exists():
#             grid_path = cand
#             break
#     if grid_path is None:
#         raise FileNotFoundError(f"Couldn't find {stem.name}.EGRID or {stem.name}.GRID next to the UNRST file.")

#     grid = EclGrid(str(grid_path))

#     # Validate SWAT keyword
#     if "SWAT" not in available_keys or len(unrst["SWAT"]) == 0:
#         raise KeyError("SWAT keyword is missing or empty in the UNRST file.")

#     nactive = len(unrst["SWAT"][0])
#     ijk = np.empty((nactive, 3), dtype=np.int32)
#     for ai in range(nactive):
#         i0, j0, k0 = grid.get_ijk(active_index=ai)
#         ijk[ai] = (i0 + 1, j0 + 1, k0 + 1)

#     # Helper for safe keyword fetch
#     def get_kw_step(kw_name, step_idx, n_target):
#         if kw_name not in available_keys:
#             return np.zeros(n_target, dtype=float)
#         kw_all = unrst[kw_name]
#         if step_idx >= len(kw_all):
#             return np.zeros(n_target, dtype=float)
#         arr = np.asarray(kw_all[step_idx], dtype=float)
#         if arr.size == n_target:
#             return arr
#         elif arr.size > n_target:
#             return arr[:n_target]
#         else:
#             out = np.zeros(n_target, dtype=float)
#             out[:arr.size] = arr
#             return out

#     # Build dataframe
#     frames = []
#     for step_idx, date in enumerate(dates):
#         if step_idx >= len(unrst["SWAT"]):
#             continue

#         swat = np.asarray(unrst["SWAT"][step_idx], dtype=float)
#         n = swat.size
#         ijk_step = ijk[:n]
#         oilkr = get_kw_step("OILKR", step_idx, n)

#         df_step = pd.DataFrame({
#             "I": ijk_step[:, 0],
#             "J": ijk_step[:, 1],
#             "K": ijk_step[:, 2],
#             "SWAT": np.round(swat, 5),
#             "OILKR": np.round(oilkr, 5),
#             "Group Index": step_idx + 1,
#             "Group Description": date.strftime("%d.%b %Y"),
#         })
#         frames.append(df_step)

#     df = pd.concat(frames, ignore_index=True)
#     df.to_csv(out_csv, index=False)
#     print(f"[INFO] Exported {len(df)} rows to {out_csv}")
    
#     # --- Add Turning Point Detection and Process Columns ---
#     df_sorted = df.sort_values(by=['I','J','K','Group Index']).reset_index(drop=True)
#     turning_points = []
#     turning_values = []
#     filled_values = []
#     process = []

#     last_turning_value = df_sorted.loc[0, 'SWAT']
#     prev_diff = 0

#     for i in range(len(df_sorted)):
#         if i == 0:
#             turning_points.append('')
#             turning_values.append('')
#             filled_values.append(last_turning_value)
#             process.append('Initial')
#             continue

#         diff = df_sorted.loc[i, 'SWAT'] - df_sorted.loc[i-1, 'SWAT']
#         if prev_diff != 0 and diff * prev_diff < 0:
#             turning_points.append('Turning Point')
#             turning_values.append(df_sorted.loc[i, 'SWAT'])
#             last_turning_value = df_sorted.loc[i, 'SWAT']
#         else:
#             turning_points.append('')
#             turning_values.append('')
#         filled_values.append(last_turning_value)

#         # Process logic
#         if diff > 0:
#             process.append('Imbibition')
#         elif diff < 0:
#             process.append('Drainage')
#         else:
#             process.append(process[-1])

#         prev_diff = diff

#     df_sorted['Turning Point'] = turning_points
#     df_sorted['Turning Point Value'] = turning_values
#     df_sorted['Turning Point Value Filled'] = filled_values
#     df_sorted['Process'] = process

#     # Save updated CSV
#     df_sorted.to_csv(out_csv, index=False)
#     print(f"[INFO] Added turning point and process columns. Saved to {out_csv}")
#     return df_sorted



# def compare_csv_plots(files, legends, output_dir="comparison_plots", cells=None):
#     """
#     Compare SWAT vs OILKR trends for multiple CSV files and save plots.
#     Filters by I index only (J=1, K=1 assumed).
#     Adds:
#       - Global color legend for timesteps (same color for same date across all files).
#       - File legend for line styles.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     dfs = [pd.read_csv(f).sort_values(by=['Group Index']) for f in files]

#     # Collect all unique dates globally
#     all_dates = sorted(set().union(*[df['Group Description'].unique() for df in dfs]))

#     # Use qualitative colormap for better differentiation
#     cmap = cm.get_cmap('tab20', len(all_dates))
#     date_color_map = {d: cmap(i) for i, d in enumerate(all_dates)}

#     # Determine cells to plot (I indices only)
#     if cells is None:
#         all_cells = set()
#         for df in dfs:
#             all_cells.update(df['I'].unique())
#         cells = sorted(all_cells)[:10]

#     markers = ['o', 'x', '^', 's']

#     for cell_i in cells:
#         plt.figure(figsize=(12, 7))

#         # Plot data for each file
#         for idx, df in enumerate(dfs):
#             # Filter by I only, J=1 and K=1
#             cell_data = df[(df['I'] == cell_i) & (df['J'] == 1) & (df['K'] == 1)]
#             if cell_data.empty:
#                 continue

#             # Line plot for file
#             plt.plot(cell_data['SWAT'], cell_data['OILKR'],
#                      linewidth=2, label=legends[idx])

#             # Scatter points with global color by date
#             for _, row in cell_data.iterrows():
#                 plt.scatter(row['SWAT'], row['OILKR'],
#                             color=date_color_map[row['Group Description']],
#                             s=80, marker=markers[idx])

#         # Title and axis labels
#         plt.title(f"SWAT vs OILKR for Cell I={cell_i}, J=1, K=1")
#         plt.xlabel("SWAT")
#         plt.ylabel("OILKR")

#         # File legend (inside plot)
#         file_legend = plt.legend(loc='upper left', title="Models")

#         # Add global date color legend (outside plot)
#         legend_patches = [mpatches.Patch(color=color, label=date) for date, color in date_color_map.items()]
#         plt.gca().add_artist(file_legend)  # Keep file legend
#         plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title="Dates")

#         plt.tight_layout()
#         plt.savefig(f"{output_dir}/cell_I{cell_i}_J1_K1.png", dpi=300, bbox_inches='tight')
#         plt.close()

#     print(f"[INFO] Plots saved in {output_dir}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compare SWAT vs OILKR from up to 4 UNRST files.")
#     parser.add_argument("file1", help="First UNRST file")
#     parser.add_argument("file2", help="Second UNRST file")
#     parser.add_argument("file3", nargs="?", default=None, help="Optional third UNRST file")
#     parser.add_argument("file4", nargs="?", default=None, help="Optional fourth UNRST file")
#     parser.add_argument("--legends", nargs="+", help="Custom legend labels for files", required=False)
#     parser.add_argument("--cells", nargs="+", type=int, help="List of I indices to plot")
#     parser.add_argument("--outdir", default="comparison_plots", help="Output directory for plots")

#     args = parser.parse_args()

#     # Output CSV names
#     csv_files = []
#     unrst_files = [args.file1, args.file2, args.file3, args.file4]
#     for i, f in enumerate(unrst_files):
#         if f:
#             csv_name = f"swat_oilkr{i+1}.csv"
#             export_unrst_to_csv(f, csv_name)
#             csv_files.append(csv_name)

#     # Legends
#     if args.legends:
#         legends = args.legends
#         if len(legends) != len(csv_files):
#             raise ValueError("Number of legends must match number of files.")
#     else:
#         legends = [f"File {i+1}" for i in range(len(csv_files))]

#     # Cells (I indices only)
#     cells = args.cells if args.cells else None

#     # Compare and plot
#     compare_csv_plots(csv_files, legends, output_dir=args.outdir, cells=cells)

    
# import argparse
# from pathlib import Path
# from ecl.eclfile import EclFile
# from ecl.grid import EclGrid
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.patches as mpatches
# import os


# def export_unrst_to_csv(unrst_file: str, output_csv: str):
#     """
#     Extract SWAT and OILKR data from an Eclipse UNRST file and export to CSV.
#     """
#     unrst_path = Path(unrst_file)
#     out_csv = Path(output_csv)

#     # Load restart file
#     unrst = EclFile(str(unrst_path))
#     dates = unrst.dates
#     available_keys = set(unrst.keys())
#     print(f"[INFO] Available keywords in {unrst_file}: {sorted(available_keys)}")

#     # Locate grid (EGRID preferred)
#     stem = unrst_path.with_suffix("")
#     grid_path = None
#     for ext in (".EGRID", ".GRID"):
#         cand = stem.with_suffix(ext)
#         if cand.exists():
#             grid_path = cand
#             break
#     if grid_path is None:
#         raise FileNotFoundError(f"Couldn't find {stem.name}.EGRID or {stem.name}.GRID next to the UNRST file.")

#     grid = EclGrid(str(grid_path))

#     # Validate SWAT keyword
#     if "SWAT" not in available_keys or len(unrst["SWAT"]) == 0:
#         raise KeyError("SWAT keyword is missing or empty in the UNRST file.")

#     nactive = len(unrst["SWAT"][0])
#     ijk = np.empty((nactive, 3), dtype=np.int32)
#     for ai in range(nactive):
#         i0, j0, k0 = grid.get_ijk(active_index=ai)
#         ijk[ai] = (i0 + 1, j0 + 1, k0 + 1)

#     # Helper for safe keyword fetch
#     def get_kw_step(kw_name, step_idx, n_target):
#         if kw_name not in available_keys:
#             return np.zeros(n_target, dtype=float)
#         kw_all = unrst[kw_name]
#         if step_idx >= len(kw_all):
#             return np.zeros(n_target, dtype=float)
#         arr = np.asarray(kw_all[step_idx], dtype=float)
#         if arr.size == n_target:
#             return arr
#         elif arr.size > n_target:
#             return arr[:n_target]
#         else:
#             out = np.zeros(n_target, dtype=float)
#             out[:arr.size] = arr
#             return out

#     # Build dataframe
#     frames = []
#     for step_idx, date in enumerate(dates):
#         if step_idx >= len(unrst["SWAT"]):
#             continue

#         swat = np.asarray(unrst["SWAT"][step_idx], dtype=float)
#         n = swat.size
#         ijk_step = ijk[:n]
#         oilkr = get_kw_step("OILKR", step_idx, n)

#         df_step = pd.DataFrame({
#             "I": ijk_step[:, 0],
#             "J": ijk_step[:, 1],
#             "K": ijk_step[:, 2],
#             "SWAT": np.round(swat, 5),
#             "OILKR": np.round(oilkr, 5),
#             "Group Index": step_idx + 1,
#             "Group Description": date.strftime("%d.%b %Y"),
#         })
#         frames.append(df_step)

#     df = pd.concat(frames, ignore_index=True)
#     df.to_csv(out_csv, index=False)
#     print(f"[INFO] Exported {len(df)} rows to {out_csv}")
    
#     # --- Turning Point Detection and Propagation for SWAT ---
#     df_sorted = df.sort_values(by=['I','J','K','Group Index']).reset_index(drop=True)
#     turning_points = []
#     turning_values = []
#     filled_values = []

#     # Initialize with first row as initial turning point
#     last_turning_value = df_sorted.loc[0, 'SWAT']
#     prev_diff = 0

#     for i in range(len(df_sorted)):
#         if i == 0:
#             turning_points.append('')
#             turning_values.append('')
#             filled_values.append(last_turning_value)
#             continue

#         diff = df_sorted.loc[i, 'SWAT'] - df_sorted.loc[i-1, 'SWAT']
#         if prev_diff != 0 and diff * prev_diff < 0:
#             # Turning point detected
#             turning_points.append('Turning Point')
#             turning_values.append(df_sorted.loc[i, 'SWAT'])
#             last_turning_value = df_sorted.loc[i, 'SWAT']
#         else:
#             turning_points.append('')
#             turning_values.append('')
#         filled_values.append(last_turning_value)
#         prev_diff = diff

#     # Add columns to DataFrame
#     df_sorted['Turning Point'] = turning_points
#     df_sorted['Turning Point Value'] = turning_values
#     df_sorted['Turning Point Value Filled'] = filled_values

#     # Save updated CSV
#     df_sorted.to_csv(out_csv, index=False)
#     print(f"[INFO] Added turning point columns and saved to {out_csv}")

#     return df


# def compare_csv_plots(files, legends, output_dir="comparison_plots", cells=None):
#     """
#     Compare SWAT vs OILKR trends for multiple CSV files and save plots.
#     Filters by I index only (J=1, K=1 assumed).
#     Adds:
#       - Global color legend for timesteps (same color for same date across all files).
#       - File legend for line styles.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     dfs = [pd.read_csv(f).sort_values(by=['Group Index']) for f in files]

#     # Collect all unique dates globally
#     all_dates = sorted(set().union(*[df['Group Description'].unique() for df in dfs]))

#     # Use qualitative colormap for better differentiation
#     cmap = cm.get_cmap('tab20', len(all_dates))
#     date_color_map = {d: cmap(i) for i, d in enumerate(all_dates)}

#     # Determine cells to plot (I indices only)
#     if cells is None:
#         all_cells = set()
#         for df in dfs:
#             all_cells.update(df['I'].unique())
#         cells = sorted(all_cells)[:10]

#     markers = ['o', 'x', '^', 's']

#     for cell_i in cells:
#         plt.figure(figsize=(12, 7))

#         # Plot data for each file
#         for idx, df in enumerate(dfs):
#             # Filter by I only, J=1 and K=1
#             cell_data = df[(df['I'] == cell_i) & (df['J'] == 1) & (df['K'] == 1)]
#             if cell_data.empty:
#                 continue

#             # Line plot for file
#             plt.plot(cell_data['SWAT'], cell_data['OILKR'],
#                      linewidth=2, label=legends[idx])

#             # Scatter points with global color by date
#             for _, row in cell_data.iterrows():
#                 plt.scatter(row['SWAT'], row['OILKR'],
#                             color=date_color_map[row['Group Description']],
#                             s=80, marker=markers[idx])

#         # Title and axis labels
#         plt.title(f"SWAT vs OILKR for Cell I={cell_i}, J=1, K=1")
#         plt.xlabel("SWAT")
#         plt.ylabel("OILKR")

#         # File legend (inside plot)
#         file_legend = plt.legend(loc='upper left', title="Models")

#         # Add global date color legend (outside plot)
#         legend_patches = [mpatches.Patch(color=color, label=date) for date, color in date_color_map.items()]
#         plt.gca().add_artist(file_legend)  # Keep file legend
#         plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title="Dates")

#         plt.tight_layout()
#         plt.savefig(f"{output_dir}/cell_I{cell_i}_J1_K1.png", dpi=300, bbox_inches='tight')
#         plt.close()

#     print(f"[INFO] Plots saved in {output_dir}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compare SWAT vs OILKR from up to 4 UNRST files.")
#     parser.add_argument("file1", help="First UNRST file")
#     parser.add_argument("file2", help="Second UNRST file")
#     parser.add_argument("file3", nargs="?", default=None, help="Optional third UNRST file")
#     parser.add_argument("file4", nargs="?", default=None, help="Optional fourth UNRST file")
#     parser.add_argument("--legends", nargs="+", help="Custom legend labels for files", required=False)
#     parser.add_argument("--cells", nargs="+", type=int, help="List of I indices to plot")
#     parser.add_argument("--outdir", default="comparison_plots", help="Output directory for plots")

#     args = parser.parse_args()

#     # Output CSV names
#     csv_files = []
#     unrst_files = [args.file1, args.file2, args.file3, args.file4]
#     for i, f in enumerate(unrst_files):
#         if f:
#             csv_name = f"swat_oilkr{i+1}.csv"
#             export_unrst_to_csv(f, csv_name)
#             csv_files.append(csv_name)

#     # Legends
#     if args.legends:
#         legends = args.legends
#         if len(legends) != len(csv_files):
#             raise ValueError("Number of legends must match number of files.")
#     else:
#         legends = [f"File {i+1}" for i in range(len(csv_files))]

#     # Cells (I indices only)
#     cells = args.cells if args.cells else None

#     # Compare and plot
#     compare_csv_plots(csv_files, legends, output_dir=args.outdir, cells=cells)

# # import argparse
# # from pathlib import Path
# # from ecl.eclfile import EclFile
# # from ecl.grid import EclGrid
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.cm as cm
# # import matplotlib.patches as mpatches
# # import os


# # def export_unrst_to_csv(unrst_file: str, output_csv: str):
# #     """
# #     Extract SWAT and OILKR data from an Eclipse UNRST file and export to CSV.
# #     """
# #     unrst_path = Path(unrst_file)
# #     out_csv = Path(output_csv)

# #     # Load restart file
# #     unrst = EclFile(str(unrst_path))
# #     dates = unrst.dates
# #     available_keys = set(unrst.keys())
# #     print(f"[INFO] Available keywords in {unrst_file}: {sorted(available_keys)}")

# #     # Locate grid (EGRID preferred)
# #     stem = unrst_path.with_suffix("")
# #     grid_path = None
# #     for ext in (".EGRID", ".GRID"):
# #         cand = stem.with_suffix(ext)
# #         if cand.exists():
# #             grid_path = cand
# #             break
# #     if grid_path is None:
# #         raise FileNotFoundError(f"Couldn't find {stem.name}.EGRID or {stem.name}.GRID next to the UNRST file.")

# #     grid = EclGrid(str(grid_path))

# #     # Validate SWAT keyword
# #     if "SWAT" not in available_keys or len(unrst["SWAT"]) == 0:
# #         raise KeyError("SWAT keyword is missing or empty in the UNRST file.")

# #     nactive = len(unrst["SWAT"][0])
# #     ijk = np.empty((nactive, 3), dtype=np.int32)
# #     for ai in range(nactive):
# #         i0, j0, k0 = grid.get_ijk(active_index=ai)
# #         ijk[ai] = (i0 + 1, j0 + 1, k0 + 1)

# #     # Helper for safe keyword fetch
# #     def get_kw_step(kw_name, step_idx, n_target):
# #         if kw_name not in available_keys:
# #             return np.zeros(n_target, dtype=float)
# #         kw_all = unrst[kw_name]
# #         if step_idx >= len(kw_all):
# #             return np.zeros(n_target, dtype=float)
# #         arr = np.asarray(kw_all[step_idx], dtype=float)
# #         if arr.size == n_target:
# #             return arr
# #         elif arr.size > n_target:
# #             return arr[:n_target]
# #         else:
# #             out = np.zeros(n_target, dtype=float)
# #             out[:arr.size] = arr
# #             return out

# #     # Build dataframe
# #     frames = []
# #     for step_idx, date in enumerate(dates):
# #         if step_idx >= len(unrst["SWAT"]):
# #             continue

# #         swat = np.asarray(unrst["SWAT"][step_idx], dtype=float)
# #         n = swat.size
# #         ijk_step = ijk[:n]
# #         oilkr = get_kw_step("OILKR", step_idx, n)

# #         df_step = pd.DataFrame({
# #             "I": ijk_step[:, 0],
# #             "J": ijk_step[:, 1],
# #             "K": ijk_step[:, 2],
# #             "SWAT": np.round(swat, 5),
# #             "OILKR": np.round(oilkr, 5),
# #             "Group Index": step_idx + 1,
# #             "Group Description": date.strftime("%d.%b %Y"),
# #         })
# #         frames.append(df_step)

# #     df = pd.concat(frames, ignore_index=True)
# #     df.to_csv(out_csv, index=False)
# #     print(f"[INFO] Exported {len(df)} rows to {out_csv}")
# #     return df


# # def compare_csv_plots(files, legends, output_dir="comparison_plots", cells=None):
# #     """
# #     Compare SWAT vs OILKR trends for multiple CSV files and save plots.
# #     Filters by I index only (J=1, K=1 assumed).
# #     Adds:
# #       - Global color legend for timesteps (same color for same date across all files).
# #       - File legend for line styles.
# #     """
# #     os.makedirs(output_dir, exist_ok=True)

# #     dfs = [pd.read_csv(f).sort_values(by=['Group Index']) for f in files]

# #     # Collect all unique dates globally
# #     all_dates = sorted(set().union(*[df['Group Description'].unique() for df in dfs]))

# #     # Use qualitative colormap for better differentiation
# #     cmap = cm.get_cmap('tab20', len(all_dates))
# #     date_color_map = {d: cmap(i) for i, d in enumerate(all_dates)}

# #     # Determine cells to plot (I indices only)
# #     if cells is None:
# #         all_cells = set()
# #         for df in dfs:
# #             all_cells.update(df['I'].unique())
# #         cells = sorted(all_cells)[:10]

# #     markers = ['o', 'x', '^', 's']

# #     for cell_i in cells:
# #         plt.figure(figsize=(12, 7))

# #         # Plot data for each file
# #         for idx, df in enumerate(dfs):
# #             # Filter by I only, J=1 and K=1
# #             cell_data = df[(df['I'] == cell_i) & (df['J'] == 1) & (df['K'] == 1)]
# #             if cell_data.empty:
# #                 continue

# #             # Line plot for file
# #             plt.plot(cell_data['SWAT'], cell_data['OILKR'],
# #                      linewidth=2, label=legends[idx])

# #             # Scatter points with global color by date
# #             for _, row in cell_data.iterrows():
# #                 plt.scatter(row['SWAT'], row['OILKR'],
# #                             color=date_color_map[row['Group Description']],
# #                             s=80, marker=markers[idx])

# #         # Title and axis labels
# #         plt.title(f"SWAT vs OILKR for Cell I={cell_i}, J=1, K=1")
# #         plt.xlabel("SWAT")
# #         plt.ylabel("OILKR")

# #         # File legend (inside plot)
# #         file_legend = plt.legend(loc='upper left', title="Models")

# #         # Add global date color legend (outside plot)
# #         legend_patches = [mpatches.Patch(color=color, label=date) for date, color in date_color_map.items()]
# #         plt.gca().add_artist(file_legend)  # Keep file legend
# #         plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title="Dates")

# #         plt.tight_layout()
# #         plt.savefig(f"{output_dir}/cell_I{cell_i}_J1_K1.png", dpi=300, bbox_inches='tight')
# #         plt.close()

# #     print(f"[INFO] Plots saved in {output_dir}")


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="Compare SWAT vs OILKR from up to 4 UNRST files.")
# #     parser.add_argument("file1", help="First UNRST file")
# #     parser.add_argument("file2", help="Second UNRST file")
# #     parser.add_argument("file3", nargs="?", default=None, help="Optional third UNRST file")
# #     parser.add_argument("file4", nargs="?", default=None, help="Optional fourth UNRST file")
# #     parser.add_argument("--legends", nargs="+", help="Custom legend labels for files", required=False)
# #     parser.add_argument("--cells", nargs="+", type=int, help="List of I indices to plot")
# #     parser.add_argument("--outdir", default="comparison_plots", help="Output directory for plots")

# #     args = parser.parse_args()

# #     # Output CSV names
# #     csv_files = []
# #     unrst_files = [args.file1, args.file2, args.file3, args.file4]
# #     for i, f in enumerate(unrst_files):
# #         if f:
# #             csv_name = f"swat_oilkr{i+1}.csv"
# #             export_unrst_to_csv(f, csv_name)
# #             csv_files.append(csv_name)

# #     # Legends
# #     if args.legends:
# #         legends = args.legends
# #         if len(legends) != len(csv_files):
# #             raise ValueError("Number of legends must match number of files.")
# #     else:
# #         legends = [f"File {i+1}" for i in range(len(csv_files))]

# #     # Cells (I indices only)
# #     cells = args.cells if args.cells else None

# #     # Compare and plot
# #     compare_csv_plots(csv_files, legends, output_dir=args.outdir, cells=cells)