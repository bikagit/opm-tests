# SPDX-FileCopyrightText: 2024 NORCE
# SPDX-License-Identifier: GPL-3.0

"""
Script to run OPM Flow and plot results for different linear solver tolerance values.
"""

import os
import csv
import math
import argparse
import itertools
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

font = {"family": "normal", "weight": "normal", "size": 20}
matplotlib.rc("font", **font)
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "monospace",
        "legend.columnspacing": 0.9,
        "legend.handlelength": 3.5,
        "legend.fontsize": 15,
        "lines.linewidth": 4,
        "axes.titlesize": 20,
        "axes.grid": True,
        "figure.figsize": (10, 10),
    }
)

# ---------------------------------------------------------------------------
# Constants — edit these or override via a config file / environment vars
# ---------------------------------------------------------------------------

np.random.seed(18)

VARIABLE = "--linear-solver-reduction="

# VALUES = [0.00414663, 0.00545092, 0.00209259, 0.00836344, 0.0023299,  0.00324877,
#  0.00400508, 0.00110894, 0.00768728, 0.00974525,1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2]
# VALUES = [ 5e-3]
# # VALUES = [ 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2,1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 5e-1, 5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 2.5e-1, 2.5e-2, 2.5e-3, 2.5e-4]
VALUES = [
    5e-3,1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
    2.5e-4, 5e-4, 1e-3, 2.5e-3,
    1e-2, 2.5e-2, 5e-2, 2.5e-1, 5e-1,
]

# VALUES = [5e-3, 1e-2, 1e-3, 1e-4, 1e-5]
DEFAULTADAPTIVE = 5e-3

NMPIS = 1
NEWTONMAXIT = 20
CNV = 1e-2
MB = 1e-7
BETA = 1
ALPHA = 0.33  # BETA * I_newton + ALPHA * I_linear

CASE = "NORNE_ATW2013"

# Path to the Flow binary — override with the FLOW_BIN environment variable
FLOW_BIN = os.environ.get(
    "FLOW_BIN",
    "../../build/opm-simulators/bin/flow",  # replace or set $FLOW_BIN
)

BASE_FLAGS = (
    " --use-best-residual=true"
    " --relaxed-max-pv-fraction=0"
    " --output-extra-convergence-info=steps,iterations"
    " --enable-ecl-output=0"
    " --use-gmres=0"
    " --linear-solver=ilu0"
    " --full-time-step-initially=1"
    f" --tolerance-cnv-relaxed={CNV}"
    f" --tolerance-cnv={CNV}"
    " --tolerance-mb=1e-7"
    " --tolerance-mb-relaxed=1e-7"
    " --newton-min-iterations=1"
    f" --newton-max-iterations={NEWTONMAXIT}"
)
# 
ADAPTIVE_FLAGS = (
    " --use-m-lmethods-tols=true"
    " --use-best-path=false"
    " --use-best-residual=true"
    " --relaxed-max-pv-fraction=0"
    " --output-extra-convergence-info=steps,iterations"
    " --enable-ecl-output=0"
    " --linear-solver=ilu0"
    " --full-time-step-initially=1"
    f" --tolerance-cnv-relaxed={CNV}"
    f" --tolerance-cnv={CNV}"
    " --tolerance-mb=1e-7"
    " --tolerance-mb-relaxed=1e-7"
    " --newton-min-iterations=1"
    f" --newton-max-iterations={NEWTONMAXIT}"
)

_BASE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_BASE_LINESTYLES = [
    "--", (0, (1, 1)), "-.", (0, (1, 10)), (0, (5, 5)),
    (5, (10, 3)), (0, (5, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, ()),
]
COLORS = list(itertools.islice(itertools.cycle(_BASE_COLORS), 62)) + ["r", "k"]
LINESTYLES = list(itertools.islice(itertools.cycle(_BASE_LINESTYLES), 63)) + ["-"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Application of ML for OPM Flow tolerances",
    )
    parser.add_argument(
        "-z", "--outpucsv", default=1, type=int,
        help="Output in table form csv (1) or NN format (0). Default: 1.",
    )
    parser.add_argument(
        "-e", "--runensemble", default=0, type=int,
        help="Run the ensemble (1=yes, 0=no). Default: 0.",
    )
    parser.add_argument(
        "-a", "--runadaptive", default=0, type=int,
        help="Run the adaptive approach (1=yes, 0=no). Default: 0.",
    )
    parser.add_argument(
        "-o", "--output", default="output",
        help="Name of output folder. Default: 'output'.",
    )
    return vars(parser.parse_known_args()[0])


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------

def _build_command(value, sim_index, out_dir, flags, adaptive=False):
    """Return the subprocess argument list for a single Flow run."""
    out_subdir = out_dir / ("adaptive" if adaptive else f"sim_{sim_index}")
    args = [
        FLOW_BIN,
        CASE,
        *flags.split(),
        f"{VARIABLE}{value}",
        f"--output-dir={out_subdir}",
    ]
    if NMPIS > 1:
        args = ["mpirun", "-np", str(NMPIS)] + args
    return args


def run_ensemble(out_dir):
    """Run all VALUES simulations, batching NPRUNS at a time in parallel."""
    npruns = 7
    n_batches = math.floor(len(VALUES) / npruns)

    for batch in range(n_batches):
        procs = []
        for j in range(npruns):
            idx = npruns * batch + j
            cmd = _build_command(VALUES[idx], idx, out_dir, BASE_FLAGS)
            procs.append(subprocess.Popen(cmd))
        for p in procs:
            p.wait()

    # Handle any remaining simulations
    finished = npruns * n_batches
    remaining_vals = VALUES[finished:]
    if remaining_vals:
        procs = []
        for i, val in enumerate(remaining_vals):
            cmd = _build_command(val, finished + i, out_dir, BASE_FLAGS)
            procs.append(subprocess.Popen(cmd))
        for p in procs:
            p.wait()


def run_adaptive(out_dir):
    """Run a single adaptive simulation."""
    cmd = _build_command(DEFAULTADAPTIVE, None, out_dir, ADAPTIVE_FLAGS, adaptive=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------------

def _parse_space_delimited_csv(filepath):
    """Read a space-delimited CSV file, returning (header_list, rows)."""
    header, rows = [], []
    with open(filepath, "r", encoding="utf8") as fh:
        for j, row in enumerate(csv.reader(fh)):
            fields = row[0].strip().split()
            if j == 0:
                header = fields
            else:
                rows.append(fields)
    return header, rows


def read_results(out_dir):
    """
    Read .INFOITER and .INFOSTEP files for each simulation.

    Returns
    -------
    info_ite      : list of rows per simulation (plus adaptive at end if present)
    info_itenam   : column names for .INFOITER
    newtit        : total Newton iterations per simulation
    linit         : total linear iterations per simulation
    finalstep     : last reported time step size (days)
    has_adaptive  : bool — whether adaptive results were found
    newton_adaptive, linear_adaptive : iteration totals for adaptive run
    """
    info_ite, info_itenam = [], []
    newtit, linit = [], []
    finalstep = 0.0

    for i, _ in enumerate(VALUES):
        sim_dir = out_dir / f"sim_{i}"
        try:
            header, rows = _parse_space_delimited_csv(sim_dir / f"{CASE}.INFOITER")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing output for sim_{i}. Did the simulation complete? "
                f"Expected: {sim_dir / f'{CASE}.INFOITER'}"
            )
        if i == 0:
            info_itenam = header
        info_ite.append(rows)

        newtit.append(0)
        linit.append(0)
        step_header, step_rows = _parse_space_delimited_csv(sim_dir / f"{CASE}.INFOSTEP")
        for row in step_rows:
            newtit[-1] += int(row[step_header.index("NewtIt")])
            linit[-1] += int(row[step_header.index("LinIt")])
        if i == 0 and step_rows:
            finalstep = float(step_rows[-1][step_header.index("TStep(day)")])

    # Optional adaptive run
    adaptive_infoiter = out_dir / "adaptive" / f"{CASE}.INFOITER"
    adaptive_infostep = out_dir / "adaptive" / f"{CASE}.INFOSTEP"
    has_adaptive = adaptive_infoiter.exists() and adaptive_infostep.exists()
    newton_adaptive, linear_adaptive = 0, 0

    if has_adaptive:
        header, rows = _parse_space_delimited_csv(adaptive_infoiter)
        info_itenam = header  # refresh in case it differs
        info_ite.append(rows)

        step_header, step_rows = _parse_space_delimited_csv(
            out_dir / "adaptive" / f"{CASE}.INFOSTEP"
        )
        for row in step_rows:
            newton_adaptive += int(row[step_header.index("NewtIt")])
            linear_adaptive += int(row[step_header.index("LinIt")])

    return (
        info_ite, info_itenam,
        newtit, linit, finalstep,
        has_adaptive, newton_adaptive, linear_adaptive,
    )


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def process_data(info_ite, info_itenam, finalstep):
    """
    Organise per-step, per-iteration residuals from raw row data.

    Returns iters, maxress, rescnvo, rescnvw, rescnvg, times, tsteps.
    """
    no_steps = int(info_ite[0][-1][info_itenam.index("ReportStep")]) + 1

    # Pre-group rows by (ReportStep, TimeStep) for O(n) lookup
    grouped = []
    for sim_rows in info_ite:
        by_step = {}
        for row in sim_rows:
            key = (int(row[info_itenam.index("ReportStep")]),
                   int(row[info_itenam.index("TimeStep")]))
            by_step.setdefault(key, []).append(row)
        grouped.append(by_step)

    iters, maxress, rescnvo, rescnvw, rescnvg, times = [], [], [], [], [], []

    for i in range(len(info_ite)):
        iters.append([[] for _ in range(no_steps)])
        maxress.append([[] for _ in range(no_steps)])
        rescnvo.append([[] for _ in range(no_steps)])
        rescnvw.append([[] for _ in range(no_steps)])
        rescnvg.append([[] for _ in range(no_steps)])

        for n in range(no_steps):
            step_rows = grouped[i].get((n, 0), [])[:NEWTONMAXIT]
            for count, row in enumerate(step_rows):
                iters[i][n].append(int(row[info_itenam.index("Iteration")]))
                maxress[i][n].append(max(
                    float(row[info_itenam.index("CNV_Gas")]),
                    float(row[info_itenam.index("CNV_Oil")]),
                    float(row[info_itenam.index("CNV_Water")]),
                ))
                rescnvo[i][n].append(float(row[info_itenam.index("CNV_Oil")]))
                rescnvw[i][n].append(float(row[info_itenam.index("CNV_Water")]))
                rescnvg[i][n].append(float(row[info_itenam.index("CNV_Gas")]))
                if i == 0 and count == 0:
                    times.append(float(row[info_itenam.index("Time")]))

    times = np.array(times)
    tsteps = list(times[1:] - times[:-1]) + [finalstep]
    return iters, maxress, rescnvo, rescnvw, rescnvg, times, tsteps


# ---------------------------------------------------------------------------
# Best-path selection
# ---------------------------------------------------------------------------

def select_best_path(maxress, rescnvo, rescnvw, rescnvg, tsteps, output_format):
    """
    Compute best tolerance path and residual sequences.

    Returns bestpath (list of strings), bestress (list of strings),
    ppath, press.
    """
    no_steps = len(tsteps)

    if output_format == 0:
        return _select_format0(maxress, tsteps, no_steps)
    else:
        return _select_format1(maxress, rescnvo, rescnvw, rescnvg, tsteps, no_steps)


def _find_best_per_iteration(maxress, n, k):
    """Return (min_residual, best_value) across all VALUES at step n, iteration k."""
    min_ress = math.inf
    best_val = math.inf
    for i, value in enumerate(VALUES):
        if k < len(maxress[i][n]) and maxress[i][n][k] < min_ress:
            min_ress = maxress[i][n][k]
            best_val = value
    return min_ress, best_val


def _select_format0(maxress, tsteps, no_steps):
    header = [
        f"#Min residual values; NEWTONMAXIT={NEWTONMAXIT}; MB={MB}; CNV={CNV}; MPI={NMPIS}\n",
        "TStep[d],Defaulta",
        *[f",iterati{i}" for i in range(NEWTONMAXIT + 1)],
        "\n",
    ]
    bestpath, bestress = [""], list(header)
    ppath, press = [], []

    for n in range(no_steps):
        bestpath.append(f"{tsteps[n]:.2e},{DEFAULTADAPTIVE:.2e},")
        bestress.append(f"{tsteps[n]:.2e},{DEFAULTADAPTIVE:.2e},")
        ppath.append([])
        press.append([])

        for k in range(NEWTONMAXIT + 1):
            if k == 0:
                bestpath += [f"{DEFAULTADAPTIVE:.2e}", ","]
                bestress += [f"{maxress[-1][n][k]:.2e}", ","]
                ppath[-1].append(math.inf)
                press[-1].append(maxress[-1][n][k])
            else:
                min_ress, best_val = _find_best_per_iteration(maxress, n, k)
                if min_ress == math.inf:
                    del bestpath[-1]
                    del bestress[-1]
                    break
                bestpath += [f"{best_val:.2e}"]
                bestress += [f"{min_ress:.2e}"]
                ppath[-1].append(best_val)
                press[-1].append(min_ress)
                if min_ress < 1:
                    break
                bestpath += [","]
                bestress += [","]
        else:
            del bestpath[-1]
            del bestress[-1]
        bestpath += "\n"
        bestress += "\n"

    return bestpath, bestress, ppath, press


def _select_format1(maxress, rescnvo, rescnvw, rescnvg, tsteps, no_steps):
    header = [
        "TStep[d],Defaulta",
        ",bestTol",
        ",cnvminmaxresid",
        ",cnvresidoil",
        ",cnvresidwater",
        ",cnvresidgas",
        ",iterationNumber",
        "\n",
    ]
    bestpath = list(header)
    bestress = []
    ppath, press = [], []

    for n in range(no_steps):
        ppath.append([])
        press.append([])

        for k in range(NEWTONMAXIT + 1):
            if k == 0:
                row = [
                    f"{tsteps[n]:.2e}", ",",
                    f"{DEFAULTADAPTIVE:.2e}", ",",
                    f"{DEFAULTADAPTIVE:.2e}", ",",
                    f"{maxress[-1][n][k]:.2e}", ",",
                    f"{rescnvo[-1][n][k]:.2e}", ",",
                    f"{rescnvw[-1][n][k]:.2e}", ",",
                    f"{rescnvg[-1][n][k]:.2e}", ",",
                    f"{k}", "\n",
                ]
                bestpath += row
                ppath[-1].append(math.inf)
            else:
                min_ress, best_val = _find_best_per_iteration(maxress, n, k)
                if min_ress == math.inf:
                    del bestpath[-1]
                    break

                # Retrieve per-component residuals for the winning sim
                best_i = next(
                    i for i, v in enumerate(VALUES)
                    if v == best_val and k < len(maxress[i][n])
                )
                row = [
                    f"{tsteps[n]:.2e}", ",",
                    f"{DEFAULTADAPTIVE:.2e}", ",",
                    f"{best_val:.2e}", ",",
                    f"{min_ress:.2e}", ",",
                    f"{rescnvo[best_i][n][k]:.2e}", ",",
                    f"{rescnvw[best_i][n][k]:.2e}", ",",
                    f"{rescnvg[best_i][n][k]:.2e}", ",",
                    f"{k}", "\n",
                ]
                bestpath += row
                ppath[-1].append(best_val)
                if min_ress < 1:
                    break
        else:
            del bestpath[-1]
        bestpath += "\n"

    return bestpath, bestress, ppath, press


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_outputs(out_dir, bestpath, bestress):
    """Write bestpath.csv and bestress.csv to out_dir."""
    (out_dir / "bestpath.csv").write_text("".join(bestpath), encoding="utf8")
    (out_dir / "bestress.csv").write_text("".join(bestress), encoding="utf8")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    out_dir, iters, maxress, press, ppath,
    newtit, linit,
    has_adaptive, newton_adaptive, linear_adaptive,
):
    """Generate and save all plots."""
    no_steps = len(iters[0])

    if len(VALUES) < 100:
        _plot_per_step(out_dir, iters, maxress, press, has_adaptive, no_steps)

    _plot_iteration_summary(
        out_dir, newtit, linit,
        has_adaptive, newton_adaptive, linear_adaptive,
        no_steps,
    )


def _plot_per_step(out_dir, iters, maxress, press, has_adaptive, no_steps):
    figs = [plt.subplots() for _ in range(no_steps)]
    axes = [ax for _, ax in figs]
    figs = [fig for fig, _ in figs]

    for i, val in enumerate(VALUES):
        for n in range(no_steps):
            axes[n].plot(
                iters[i][n], maxress[i][n],
                color=COLORS[i],
                label=VARIABLE + f"{val}",
                ls=LINESTYLES[i],
                lw=1,
            )

    for n in range(no_steps):
        axes[n].plot(
            range(len(press[n])), press[n],
            color=COLORS[-1], label="best path", ls="dotted", lw=1,
        )

    if has_adaptive:
        for n in range(no_steps):
            axes[n].plot(
                iters[-1][n], maxress[-1][n],
                color="b", label="adaptive", ls="", marker="*", lw=3,
            )

    last_report = iters[0][-1]  # for title
    for n in range(no_steps):
        ax = axes[n]
        ax.set_ylabel("Max normalised residuals (CNV) [-]")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration no.")
        ax.set_title(f"{CASE}, report step {n} out of {no_steps - 1}")
        ax.legend(prop={"size": 12})
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        figs[n].savefig(out_dir / f"reportstep_{n}.png", bbox_inches="tight")
        plt.close(figs[n])


def _plot_iteration_summary(
    out_dir, newtit, linit,
    has_adaptive, newton_adaptive, linear_adaptive,
    no_steps,
):
    def _base_fig(ylabel):
        fig, ax = plt.subplots()
        ax.set_title(f"{CASE}, Total no. report steps {no_steps}")
        ax.set_xscale("log")
        ax.set_xlabel(VARIABLE[2:-1])
        ax.set_ylabel(ylabel)
        return fig, ax

    # Weighted cost
    fig, ax = _base_fig(
        r"$\beta$I$_{N}$+$\alpha$I$_{L}$ ("
        + r"$\beta$=" + f"{BETA}, " + r"$\alpha$=" + f"{ALPHA})"
    )
    ax.plot(VALUES, BETA * np.array(newtit) + ALPHA * np.array(linit),
            color="k", marker="*", lw=1)
    if has_adaptive:
        ax.axhline(y=BETA * newton_adaptive + ALPHA * linear_adaptive, color="b", lw=2)
    fig.savefig(out_dir / "totaliterationswighted.png", bbox_inches="tight")
    plt.close(fig)

    # Newton iterations
    fig, ax = _base_fig(r"I$_{N}$")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(VALUES, newtit, color="k", marker="*", lw=1)
    if has_adaptive:
        ax.axhline(y=newton_adaptive, color="b", lw=2)
    fig.savefig(out_dir / "newtoniterations.png", bbox_inches="tight")
    plt.close(fig)

    # Linear iterations
    fig, ax = _base_fig(r"I$_{L}$")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(VALUES, linit, color="k", marker="*", lw=1)
    if has_adaptive:
        ax.axhline(y=linear_adaptive, color="b", lw=2)
    fig.savefig(out_dir / "lineariterations.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cmdargs = parse_args()
    out_dir = Path.cwd() / cmdargs["output"].strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    output_format = cmdargs["outpucsv"]

    if cmdargs["runensemble"] == 1:
        run_ensemble(out_dir)

    if cmdargs["runadaptive"] == 1:
        run_adaptive(out_dir)

    (
        info_ite, info_itenam,
        newtit, linit, finalstep,
        has_adaptive, newton_adaptive, linear_adaptive,
    ) = read_results(out_dir)

    iters, maxress, rescnvo, rescnvw, rescnvg, times, tsteps = process_data(
        info_ite, info_itenam, finalstep
    )

    bestpath, bestress, ppath, press = select_best_path(
        maxress, rescnvo, rescnvw, rescnvg, tsteps, output_format
    )

    write_outputs(out_dir, bestpath, bestress)

    plot_results(
        out_dir, iters, maxress, press, ppath,
        newtit, linit,
        has_adaptive, newton_adaptive, linear_adaptive,
    )


if __name__ == "__main__":
    main()