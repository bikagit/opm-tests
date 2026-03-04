
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from ecl.eclfile import EclFile
from ecl.grid import EclGrid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import os
import re


def _parse_color_token(token: str):
    """
    Accepts:
    - Named colors (e.g., 'red', 'tab:blue')
    - Hex '#rrggbb' or '#rrggbbaa'
    - Comma-separated floats 'r,g,b' or 'r,g,b,a' in [0,1]
    Returns a Matplotlib-compatible RGBA tuple or raises ValueError.
    """
    t = token.strip()

    # RGB(A) tuple as comma-separated floats
    if "," in t:
        parts = [p.strip() for p in t.split(",")]
        vals = [float(p) for p in parts]
        if not (3 <= len(vals) <= 4):
            raise ValueError(f"Invalid RGB(A) tuple: {token}")
        if any((v < 0.0 or v > 1.0) for v in vals):
            raise ValueError(f"RGB(A) values must be in [0,1]: {token}")
        return tuple(vals)

    # Named or hex — let Matplotlib validate
    try:
        return mcolors.to_rgba(t)
    except ValueError:
        raise ValueError(f"Invalid color specifier: {token}")


def export_unrst_to_csv(unrst_file: str, output_csv: str, selected_dates=None):
    """
    Extract SWAT and OILKR data from an Eclipse UNRST file and export to CSV.
    Filters timesteps based on selected_dates if provided.
    Also appends turning point/process columns for SWAT.
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
        raise FileNotFoundError(
            f"Couldn't find {stem.name}.EGRID or {stem.name}.GRID next to the UNRST file."
        )

    grid = EclGrid(str(grid_path))

    # Validate SWAT keyword
    if "SWAT" not in available_keys or len(unrst["SWAT"]) == 0:
        raise KeyError("SWAT keyword is missing or empty in the UNRST file.")

    nactive = len(unrst["SWAT"][0])

    # Build IJK map for active cells (1-based indices)
    ijk = np.empty((nactive, 3), dtype=np.int32)
    for ai in range(nactive):
        i0, j0, k0 = grid.get_ijk(active_index=ai)
        ijk[ai] = (i0 + 1, j0 + 1, k0 + 1)

    # Helper for safe keyword fetch (returns zeros if missing or mismatched size)
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

    # Extract data into dataframe
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

        df_step = pd.DataFrame(
            {
                "I": ijk_step[:, 0],
                "J": ijk_step[:, 1],
                "K": ijk_step[:, 2],
                "SWAT": np.round(swat, 5),
                "OILKR": np.round(oilkr, 5),
                "Group Index": step_idx + 1,
                "Group Description": date_str,
            }
        )
        frames.append(df_step)

    if not frames:
        raise ValueError("No data found for the specified dates.")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Exported {len(df)} rows to {out_csv}")

    # --- Add Turning Point Detection and Process Columns ---
    df_sorted = df.sort_values(by=["I", "J", "K", "Group Index"]).reset_index(drop=True)

    turning_points = []
    turning_values = []
    filled_values = []
    process = []

    last_turning_value = df_sorted.loc[0, "SWAT"]
    prev_diff = 0.0

    for i in range(len(df_sorted)):
        if i == 0:
            turning_points.append("")
            turning_values.append("")
            filled_values.append(last_turning_value)
            process.append("Initial")
            continue

        diff = df_sorted.loc[i, "SWAT"] - df_sorted.loc[i - 1, "SWAT"]
        if prev_diff != 0 and diff * prev_diff < 0:
            # Turning point
            turning_points.append("Turning Point")
            turning_values.append(df_sorted.loc[i, "SWAT"])
            last_turning_value = df_sorted.loc[i, "SWAT"]
            filled_values.append(last_turning_value)
        else:
            turning_points.append("")
            turning_values.append("")
            filled_values.append(last_turning_value)

        if diff > 0:
            process.append("Imbibition")
        elif diff < 0:
            process.append("Drainage")
        else:
            process.append(process[-1] if process else "Initial")
        prev_diff = diff

    df_sorted["Turning Point"] = turning_points
    df_sorted["Turning Point Value"] = turning_values
    df_sorted["Turning Point Value Filled"] = filled_values
    df_sorted["Process"] = process

    df_sorted.to_csv(out_csv, index=False)
    print(f"[INFO] Added turning point and process columns. Saved to {out_csv}")
    return df_sorted


# --- NEW: Map "DD.MMM YYYY" -> "Day <day-of-month>" (e.g., "04.Jan 2026" -> "Day 4")
def _to_calendar_day_label(date_str: str) -> str:
    m = re.match(r"^(\d{2})\.[A-Za-z]{3}\s+\d{4}$", date_str.strip())
    if m:
        return f"Day {int(m.group(1))}"
    # Fallback: leave as-is if we cannot parse
    return date_str


def compare_csv_plots(files, legends, output_dir="comparison_plots", cells=None, model_colors=None):
    """
    Compare SWAT vs OILKR for multiple CSV files and save plots.
    - Lines encode the model (color per model).
    - Points use facecolor for date and edgecolor for model.
    - If `model_colors` is provided, it must be a list of RGBA tuples (or
      any Matplotlib-compatible color); otherwise a default palette is used.
    """
    os.makedirs(output_dir, exist_ok=True)
    dfs = [pd.read_csv(f).sort_values(by=["Group Index"]) for f in files]

    # --- Global date colors (consistent across all files) ---
    all_dates = sorted(set().union(*[df["Group Description"].unique() for df in dfs]))

    # Map each unique date to a display label using calendar day-of-month
    date_display_map = {d: _to_calendar_day_label(d) for d in all_dates}

    # Custom list of 15 highly distinguishable colors (keep color per original date)
    custom_colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
        "#fc8d62", "#8da0cb", "#060606", "#a6d854", "#2fffb6"
    ]
    date_color_map = {d: custom_colors[i % len(custom_colors)] for i, d in enumerate(all_dates)}

    # --- Model colors: from user if provided, else default qualitative palette ---
    if model_colors is not None:
        model_colors_final = model_colors
    else:
        # Default qualitative palette
        model_cmap = cm.get_cmap('tab20c', max(7, len(dfs)))
        model_colors_final = [model_cmap(i % model_cmap.N) for i in range(len(dfs))]

    # Default cell selection: first 10 unique I-indices across all files (J=1, K=1)
    if cells is None:
        all_cells = set()
        for df in dfs:
            all_cells.update(df["I"].unique())
        cells = sorted(all_cells)[:10]

    # Marker shapes (cycle if more models than shapes)
    markers = ['o', '^', 's', 'D', 'P', '*', 'v', 'x', '<', '>']
    from matplotlib.lines import Line2D

    for cell_i in cells:
        plt.figure(figsize=(8, 5))
        for idx, df in enumerate(dfs):
            cell_data = df[(df["I"] == cell_i) & (df["J"] == 1) & (df["K"] == 1)]
            if cell_data.empty:
                continue

            # --- Line encodes MODEL (color) ---
            plt.plot(
                cell_data["SWAT"],
                cell_data["OILKR"],
                linewidth=2,
                color=model_colors_final[idx],
                label=legends[idx]
            )

            # --- Points: facecolor encodes DATE; edgecolor encodes MODEL ---
            for _, row in cell_data.iterrows():
                plt.scatter(
                    row["SWAT"],
                    row["OILKR"],
                    s=25,
                    marker=markers[idx % len(markers)],
                    facecolor=date_color_map[row["Group Description"]],
                    edgecolor=model_colors_final[idx],
                    linewidths=1.0,
                    zorder=3
                )

        # Labels
        plt.xlabel('Saturation ' r"$(S_w)$", fontsize=17)
        plt.ylabel('Rel. permeability ' r"$(k_{{rn}})$", fontsize=17)

        # (1) Models legend (lines with the model colors)
        model_handles = [
            Line2D([0], [0], color=model_colors_final[i], lw=2, label=legends[i])
            for i in range(len(dfs))
        ]
        file_legend = plt.legend(
            handles=model_handles,
            loc='lower left',
            title="Models",
            fontsize=13,  # labels in the legend
        )
        file_legend.get_title().set_fontsize(13)  # legend title

        # (2) Days legend (colored patches) — uses "Day <n>" labels
        date_patches = [
            mpatches.Patch(facecolor=color, edgecolor='none', label=date_display_map[date])
            for date, color in date_color_map.items()
        ]
        plt.gca().add_artist(file_legend)
        dates_legend = plt.legend(
            handles=date_patches,
            loc='upper right',
            title="Days",
            fontsize=13,  # labels in the legend
        )
        dates_legend.get_title().set_fontsize(13)  # legend title

        plt.tight_layout()
        outpng = f"{output_dir}/cell_I{cell_i}_J1_K1.png"
        plt.savefig(outpng, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"[INFO] Plots saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare SWAT vs OILKR from up to 4 UNRST files."
    )
    parser.add_argument("file1", help="First UNRST file")
    parser.add_argument("file2", help="Second UNRST file")
    parser.add_argument("file3", nargs="?", default=None, help="Optional third UNRST file")
    parser.add_argument("file4", nargs="?", default=None, help="Optional fourth UNRST file")

    # Allow custom CSV names, one per provided UNRST file
    parser.add_argument(
        "--csv-names",
        nargs="+",
        help="Custom CSV output names corresponding to each UNRST file (e.g., runA.csv runB.csv ...)."
    )

    parser.add_argument("--legends", nargs="+", help="Custom legend labels for files", required=False)
    parser.add_argument("--cells", nargs="+", type=int, help="List of I indices to plot")
    parser.add_argument("--outdir", default="comparison_plots", help="Output directory for plots")
    parser.add_argument("--dates", nargs="+", help="List of dates to extract (format: DD.MMM YYYY)")
    parser.add_argument("--list-dates", action="store_true", help="List all available dates and exit")

    # NEW: explicit per-model colors
    parser.add_argument(
        "--model-colors",
        nargs="+",
        help=(
            "Explicit colors for each model (same count as models). "
            "Examples: red green blue "
            "'#d62728' '#2ca02c' '#1f77b4' "
            "'0.8,0.2,0.2' '0.2,0.6,0.2' '0.2,0.4,0.8'."
        ),
    )

    args = parser.parse_args()

    # Handle --list-dates
    if args.list_dates:
        unrst_files_all = [args.file1, args.file2, args.file3, args.file4]
        for f in unrst_files_all:
            if f:
                unrst = EclFile(str(Path(f)))
                print(f"\n[INFO] Available dates in {f}:")
                for d in unrst.dates:
                    print(" ", d.strftime("%d.%b %Y"))
        exit(0)

    # Flexible date validation / normalization
    if args.dates:
        normalized_dates = []
        for d in args.dates:
            d_clean = d.strip().replace('"', '').replace("'", "")
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

    # Collect provided UNRST files (ignore None)
    unrst_files = [f for f in [args.file1, args.file2, args.file3, args.file4] if f]
    if not unrst_files:
        raise ValueError("At least one UNRST file must be provided.")

    # Validate/prepare CSV names
    if args.csv_names:
        if len(args.csv_names) != len(unrst_files):
            raise ValueError(
                f"Number of --csv-names ({len(args.csv_names)}) must match number of UNRST files ({len(unrst_files)})."
            )
        csv_files = args.csv_names
    else:
        # Fallback to auto-naming
        csv_files = [f"swat_oilkr{i+1}.csv" for i in range(len(unrst_files))]

    # Export each UNRST to the corresponding CSV
    for f, csv_name in zip(unrst_files, csv_files):
        export_unrst_to_csv(f, csv_name, selected_dates=args.dates)

    # Legends
    if args.legends:
        legends = args.legends
        if len(legends) != len(csv_files):
            raise ValueError("Number of legends must match number of files.")
    else:
        legends = [f"File {i+1}" for i in range(len(csv_files))]

    # Cells
    cells = args.cells if args.cells else None

    # Parse/validate user-provided model colors (optional)
    user_model_colors = None
    if args.model_colors:
        if len(args.model_colors) != len(csv_files):
            raise ValueError(
                f"Number of --model-colors ({len(args.model_colors)}) must match number of models ({len(csv_files)})."
            )
        parsed_colors = []
        fallback = False
        for tok in args.model_colors:
            try:
                parsed_colors.append(_parse_color_token(tok))
            except ValueError as e:
                print(f"[WARNING] {e}. Falling back to default palette for model colors.")
                fallback = True
                break
        if not fallback:
            user_model_colors = parsed_colors

    # Plot comparison
    compare_csv_plots(
        csv_files,
        legends,
        output_dir=args.outdir,
        cells=cells,
        model_colors=user_model_colors
    )
