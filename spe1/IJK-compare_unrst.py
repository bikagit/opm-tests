import argparse
from pathlib import Path
from ecl.eclfile import EclFile
from ecl.grid import EclGrid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


def export_unrst_to_csv(unrst_file: str, output_csv: str):
    """
    Extract SWAT and OILKR data from an Eclipse UNRST file and export to CSV.
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
            "Group Description": date.strftime("%d.%b %Y"),
        })
        frames.append(df_step)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Exported {len(df)} rows to {out_csv}")
    return df


def compare_csv_plots(files, legends, output_dir="comparison_plots", cells=None):
    """
    Compare SWAT vs OILKR trends for multiple CSV files and save plots.
    Supports up to 4 files with custom legend labels.
    """
    os.makedirs(output_dir, exist_ok=True)

    dfs = [pd.read_csv(f).sort_values(by=['Group Index']) for f in files]
    color_maps = [cm.viridis, cm.plasma, cm.inferno, cm.cividis]  # Different colormaps for each file

    # Map dates to colors for each file
    date_maps = []
    for i, df in enumerate(dfs):
        dates = df['Group Description'].unique()
        cmap = cm.get_cmap(color_maps[i], len(dates))
        date_maps.append({d: cmap(j) for j, d in enumerate(dates)})

    # Determine cells to plot
    if cells is None:
        all_cells = set()
        for df in dfs:
            all_cells.update(zip(df['I'], df['J'], df['K']))
        cells = sorted(all_cells)[:10]

    markers = ['o', 'x', '^', 's']

    for cell_id in cells:
        plt.figure(figsize=(8, 6))

        for idx, df in enumerate(dfs):
            cell_data = df[(df['I'] == cell_id[0]) & (df['J'] == cell_id[1]) & (df['K'] == cell_id[2])]
            if cell_data.empty:
                continue

            # Line plot
            plt.plot(cell_data['SWAT'], cell_data['OILKR'],
                     linewidth=2, label=legends[idx])

            # Scatter points with color by date
            for _, row in cell_data.iterrows():
                plt.scatter(row['SWAT'], row['OILKR'],
                            color=date_maps[idx][row['Group Description']],
                            s=80, marker=markers[idx])

        plt.title(f"SWAT vs OILKR for Cell (I={cell_id[0]}, J={cell_id[1]}, K={cell_id[2]})")
        plt.xlabel("SWAT")
        plt.ylabel("OILKR")
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cell_{cell_id[0]}_{cell_id[1]}_{cell_id[2]}.png", dpi=300)
        plt.close()

    print(f"[INFO] Plots saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SWAT vs OILKR from up to 4 UNRST files.")
    parser.add_argument("file1", help="First UNRST file")
    parser.add_argument("file2", help="Second UNRST file")
    parser.add_argument("file3", nargs="?", default=None, help="Optional third UNRST file")
    parser.add_argument("file4", nargs="?", default=None, help="Optional fourth UNRST file")
    parser.add_argument("--legends", nargs="+", help="Custom legend labels for files (e.g., Base LSLBM Variant)", required=False)
    parser.add_argument("--cells", nargs="+", type=int, help="List of cell IJK indices to plot (flattened as I J K ...)")
    parser.add_argument("--outdir", default="comparison_plots", help="Output directory for plots")

    args = parser.parse_args()

    # Output CSV names
    csv_files = []
    unrst_files = [args.file1, args.file2, args.file3, args.file4]
    for i, f in enumerate(unrst_files):
        if f:
            csv_name = f"swat_oilkr{i+1}.csv"
            export_unrst_to_csv(f, csv_name)
            csv_files.append(csv_name)

    # Legends
    if args.legends:
        legends = args.legends
        if len(legends) != len(csv_files):
            raise ValueError("Number of legends must match number of files.")
    else:
        legends = [f"File {i+1}" for i in range(len(csv_files))]

    # Cells
    if args.cells:
        if len(args.cells) % 3 != 0:
            raise ValueError("Cells must be provided as triplets: I J K ...")
        cells = [(args.cells[i], args.cells[i+1], args.cells[i+2]) for i in range(0, len(args.cells), 3)]
    else:
        cells = None

    # Compare and plot
    compare_csv_plots(csv_files, legends, output_dir=args.outdir, cells=cells)