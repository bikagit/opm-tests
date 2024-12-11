# SPDX-FileCopyrightText: 2024 NORCE
# SPDX-License-Identifier: GPL-3.0

""""
Script to run OPM Flow and plot for different values in given flags
"""

import os
import csv
import math as mt
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

np.random.seed(18)

# PARSER
parser = argparse.ArgumentParser(
    description="Application of ML for OPM Flow tolerances",
)

parser.add_argument(
    "-z",
    "--outpucsv",
    default=1,
    help="Output in table form csv ('1' by default).",
)

parser.add_argument(
    "-e",
    "--runensemble",
    default=0,
    help="Run the ensemble ('1' by default).",
)
parser.add_argument(
    "-a",
    "--runadaptive",
    default=0,
    help="Run the adaptive apporach ('0' by default).",
)
parser.add_argument(
    "-o",
    "--output",
    default="output",
    help="Name of output folder ('output' by default).",
)
cmdargs = vars(parser.parse_known_args()[0])
cwd = os.getcwd()
# INPUTS
VARIABLE = "--linear-solver-reduction="
VALUES = [5e-3, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 5e-1, 5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 2.5e-1, 2.5e-2, 2.5e-3, 2.5e-4]

#VALUES = [1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2]
# VALUES = [6e-3,5e-3]
DEFAULTADAPTIVE = 5e-3
# VALUES = np.random.uniform(1e-2,1e-3,10)
# VALUES = np.random.uniform(2e-3,8e-3,11)
OUTPUTFORMAT = int(cmdargs["outpucsv"])
RUNENSEMBLE = int(cmdargs["runensemble"])
RUNADAPTIVE = int(cmdargs["runadaptive"])
# if OUTPUTFORMAT == 0:
OUTFOL = cwd + "/" + cmdargs["output"].strip()

# if OUTPUTFORMAT == 1:
#  OUTFOL = cwd + "/" + cmdargs["outputNNformat"].strip()

NPRUNS = len(VALUES)
# NPRUNS = 11
# NPRUNS = 5

NMPIS = 1
NEWTONMAXIT = 20
CNV = 1e-2
MB = 1e-7
# FLOW = "/Users/macbookn/activopmwkspc/master/build/opm-simulators/bin/flow "
FLOW = "/Users/macbookn/hackatonwork/build/opm-simulators/bin/flow "

FLOWADAPTIVE = "/Users/macbookn/hackatonwork/build/opm-simulators/bin/flow "
CASE = "SPE10-MOD02-01"
BETA = 1
ALPHA = 0.33 # BETA * I_newton + ALPHA * I_linear (see https://opm-project.org/wp-content/uploads/2024/04/saeternes_opm_summit_230409_share.pdf) --output-mode=none 
# FLAGS = (
#     f" --newton-max-iterations={NEWTONMAXIT} --tolerance-cnv={CNV} --tolerance-cnv-relaxed={CNV} "
#     f" --tolerance-mb={MB} --tolerance-mb-relaxed={MB}
#     + "--full-time-step-initially=1 --time-step-control=newtoniterationcount "
#     + "--output-extra-convergence-info=steps,iterations "
#     + "--linear-solver=ilu0 --enable-ecl-output=0 --relaxed-max-pv-fraction=0 "
# )

#     f" --tolerance-mb={MB} --tolerance-mb-relaxed={MB}1e-5
# --tolerance-cnv-relaxed={CNV} 
FLAGS = f" --use-best-residual=true --relaxed-max-pv-fraction=0 --output-extra-convergence-info=steps,iterations --enable-ecl-output=0 --full-time-step-initially=1  --tolerance-cnv-relaxed={CNV} --tolerance-cnv={CNV} --tolerance-mb=1e-5 --tolerance-mb-relaxed=1e-5 --newton-min-iterations=1 --newton-max-iterations={NEWTONMAXIT}  "
FLAGSADAPT = f"  --use-m-lmethods-tols=true --use-best-residual=true  --relaxed-max-pv-fraction=0 --output-extra-convergence-info=steps,iterations --enable-ecl-output=0 --full-time-step-initially=1  --tolerance-cnv-relaxed={CNV} --tolerance-cnv={CNV} --tolerance-mb=1e-5 --tolerance-mb-relaxed=1e-5 --newton-min-iterations=1 --newton-max-iterations={NEWTONMAXIT} "

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "r",
    "k",
]
LINESTYLE = [
    "--",
    (0, (1, 1)),
    "-.",
    (0, (1, 10)),
    (0, (1, 1)),
    (5, (10, 3)),
    (0, (5, 10)),
    (0, (5, 5)),
    (0, (5, 1)),
    (0, (3, 10, 1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (3, 1, 1, 1)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 10, 1, 10, 1, 10)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, ()),
     "--",
    (0, (1, 1)),
    "-.",
    (0, (1, 10)),
    (0, (1, 1)),
    (5, (10, 3)),
    (0, (5, 10)),
    (0, (5, 5)),
    (0, (5, 1)),
    (0, (3, 10, 1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (3, 1, 1, 1)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 10, 1, 10, 1, 10)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, ()),
     "--",
    (0, (1, 1)),
    "-.",
    (0, (1, 10)),
    (0, (1, 1)),
    (5, (10, 3)),
    (0, (5, 10)),
    (0, (5, 5)),
    (0, (5, 1)),
    (0, (3, 10, 1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (3, 1, 1, 1)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 10, 1, 10, 1, 10)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, ()),
    "-",
]
# RUNS
if RUNENSEMBLE == 1:
    for i in range(mt.floor(len(VALUES) / NPRUNS)):
        command = ""
        for j in range(NPRUNS):
            if NMPIS == 1:
                command += (
                    FLOW
                    + CASE
                    + FLAGS
                    + VARIABLE
                    + f"{VALUES[NPRUNS*i+j]}"
                    + f" --output-dir={OUTFOL}/sim_{NPRUNS*i+j}"
                    " & "
                )
            else:
                command += (
                    f"mpirun -np {NMPIS} "
                    + FLOW
                    + CASE
                    + FLAGS
                    + VARIABLE
                    + f"{VALUES[NPRUNS*i+j]}"
                    + f" --output-dir={OUTFOL}/sim_{NPRUNS*i+j}"
                    + " & "
                )
        command += "wait"
        os.system(command)
    finished = NPRUNS * mt.floor(len(VALUES) / NPRUNS)
    remaining = len(VALUES) - finished
    command = ""
    for i in range(remaining):
        if NMPIS == 1:
            command += (
                FLOW
                + CASE
                + FLAGS
                + VARIABLE
                + str(VALUES[finished + i])
                + f" --output-dir={OUTFOL}/sim_{finished+i}"
                " & "
            )
        else:
            command += (
                f"mpirun -np {NMPIS} "
                + FLOW
                + CASE
                + FLAGS
                + VARIABLE
                + str(VALUES[finished + i])
                + f" --output-dir={OUTFOL}/sim_{finished+i}"
                + " & "
            )
    command += "wait"
    os.system(command)
if RUNADAPTIVE == 1:
    command = (
            FLOW
            + CASE
            + FLAGSADAPT
            + VARIABLE
            + f"{DEFAULTADAPTIVE}"
            + f" --output-dir={OUTFOL}/adaptive"
        )
    os.system(command)
# READ
info_ite, info_itenam = [], []
for i, val in enumerate(VALUES):
    info_ite.append([])
    with open(f"{OUTFOL}/sim_{i}/{CASE}.INFOITER", "r", encoding="utf8") as file:
        for j, row in enumerate(csv.reader(file)):
            if j == 0 and i == 0:
                info_itenam = (row[0].strip()).split()
            elif j > 0:
                info_ite[-1].append(list(column for column in (row[0].strip()).split()))
info_stepnam, finalstep, newtit, linit = [], [], [], []
for i, val in enumerate(VALUES):
    newtit.append(0)
    linit.append(0)
    with open(f"{OUTFOL}/sim_{i}/{CASE}.INFOSTEP", "r", encoding="utf8") as file:
        for j, row in enumerate(csv.reader(file)):
            if j == 0 and i == 0:
                info_stepnam = (row[0].strip()).split()
            elif j > 0:
                newtit[-1] += int((row[0].strip()).split()[info_stepnam.index("NewtIt")])
                linit[-1] += int((row[0].strip()).split()[info_stepnam.index("LinIt")])
        if i == 0:
            finalstep = float((row[0].strip()).split()[info_stepnam.index("TStep(day)")])
if os.path.exists(f"{OUTFOL}/adaptive/{CASE}.INFOITER"):
    info_ite.append([])
    with open(f"{OUTFOL}/adaptive/{CASE}.INFOITER", "r", encoding="utf8") as file:
        for j, row in enumerate(csv.reader(file)):
            if j == 0:
                info_itenam = (row[0].strip()).split()
            elif j > 0:
                info_ite[-1].append(list(column for column in (row[0].strip()).split()))
    newtita = 0
    linita = 0
    with open(f"{OUTFOL}/adaptive/{CASE}.INFOSTEP", "r", encoding="utf8") as file:
        for j, row in enumerate(csv.reader(file)):
            if j == 0:
                info_stepnam = (row[0].strip()).split()
            elif j > 0:
                newtita += int((row[0].strip()).split()[info_stepnam.index("NewtIt")])
                linita += int((row[0].strip()).split()[info_stepnam.index("LinIt")])
# PROCESS
iters, maxress, rescnvo, rescnvw, rescnvg, times = [], [], [], [], [], []
no_steps = int(info_ite[0][-1][info_itenam.index("ReportStep")]) + 1
for i in range(len(info_ite)):
    iters.append([])
    maxress.append([])
    # 
    rescnvo.append([])
    rescnvw.append([])
    rescnvg.append([])
    for n in range(no_steps):
        iters[-1].append([])
        maxress[-1].append([])
        # 
        rescnvo[-1].append([])
        rescnvw[-1].append([])
        rescnvg[-1].append([])
        count = 0
        for row in info_ite[i]:
            if int(row[info_itenam.index("ReportStep")]) == n and int(row[info_itenam.index("TimeStep")]) == 0:
                iters[-1][-1].append(int(row[info_itenam.index("Iteration")]))
                maxress[-1][-1].append(
                    max(
                        # float(row[info_itenam.index("CNV_Gas")])/CNV,
                        # float(row[info_itenam.index("CNV_Oil")])/CNV,
                        # float(row[info_itenam.index("CNV_Water")])/CNV,
                        # float(row[info_itenam.index("MB_Gas")])/MB,
                        # float(row[info_itenam.index("MB_Oil")])/MB,
                        # float(row[info_itenam.index("MB_Water")])/MB,
                        float(row[info_itenam.index("CNV_Gas")]),
                        float(row[info_itenam.index("CNV_Oil")]),
                        float(row[info_itenam.index("CNV_Water")]),
                        # float(row[info_itenam.index("MB_Gas")]),
                        # float(row[info_itenam.index("MB_Oil")]),
                        # float(row[info_itenam.index("MB_Water")]),
                    )
                )
                rescnvo[-1][-1].append( float(row[info_itenam.index("CNV_Oil")]) )
                rescnvw[-1][-1].append( float(row[info_itenam.index("CNV_Water")]) )
                rescnvg[-1][-1].append( float(row[info_itenam.index("CNV_Gas")]) )

                if count == 0 and i == 0:
                    times.append(float(row[info_itenam.index("Time")]))
                count += 1
            if count > NEWTONMAXIT or (int(row[info_itenam.index("ReportStep")]) == n and int(row[info_itenam.index("TimeStep")]) == 1):
                break
times = np.array(times)
tsteps = list(times[1:] - times[:-1]) + [finalstep]

# SELECT
if OUTPUTFORMAT == 0:
    bestress = [
        f"#Min residual values; NEWTONMAXIT={NEWTONMAXIT}; MB={MB}; CNV={CNV}; MPI={NMPIS}\n"
    ]
    bestress += ["TStep[d],Defaulta"]
    #bestpath = [
    #    f"#{VARIABLE[2:-1]} values; NEWTONMAXIT={NEWTONMAXIT}; MB={MB}; CNV={CNV}; MPI={NMPIS}\n"
    #]
    #bestpath += ["TStep[d],Defaulta"]
    for i in range(NEWTONMAXIT + 1):
        #bestpath += [f",iterati{i}"]
        bestress += [f",iterati{i}"]
    #bestpath += ["\n"]
    bestpath = [""]
    bestress += ["\n"]
    ppath = []
    press = []
    for n in range(no_steps):
        bestpath.append(f"{tsteps[n]:.2e},{DEFAULTADAPTIVE:.2e},")
        bestress.append(f"{tsteps[n]:.2e},{DEFAULTADAPTIVE:.2e},")
        ppath.append([])
        press.append([])
        bestvalue = mt.inf
        for k in range(NEWTONMAXIT+1):
            if k == 0:
                bestpath += [f"{DEFAULTADAPTIVE:.2e}"]
                bestress += [f"{maxress[-1][n][k]:.2e}"]
                ppath[-1].append(bestvalue)
                press[-1].append(maxress[-1][n][k])
                bestpath += [","]
                bestress += [","]
            else:
                minress = mt.inf
                for i, value in enumerate(VALUES):
                    if k < len(maxress[i][n]):
                        # remove the value <= bestvalue to allow free mvmt of tols
                        # if maxress[i][n][k] < minress and value <= bestvalue:  
                        if maxress[i][n][k] < minress:
                            minress = maxress[i][n][k]
                            bestvalue = value
                if minress == mt.inf:
                    del bestpath[-1]
                    del bestress[-1]
                    break
                bestpath += [f"{bestvalue:.2e}"]
                bestress += [f"{minress:.2e}"]
                ppath[-1].append(bestvalue)
                press[-1].append(minress)
                if minress < 1:
                    break
                bestpath += [","]
                bestress += [","]
        if k == NEWTONMAXIT:
            del bestpath[-1]
            del bestress[-1]
        bestpath += "\n"
        bestress += "\n"

if OUTPUTFORMAT == 1:
    # SELECT
    bestress = [
    #     f"#Min residual valuesBLAHHHHHH; NEWTONMAXIT={NEWTONMAXIT}; MB={MB}; CNV={CNV}; MPI={NMPIS}\n"
    ]
    # bestress += ["TStep[d],Defaulta"]
    bestpath = [
    #    f"#{VARIABLE[2:-1]} values; NEWTONMAXIT={NEWTONMAXIT}; MB={MB}; CNV={CNV}; MPI={NMPIS}\n"
    ]
    # bestpath += ["Step,TStep[d],Defaulta"]
    bestpath += ["TStep[d],Defaulta"]

    # bestpath += ["Step"]
    bestpath += [f",bestTol"]
    bestpath += [f",cnvminmaxresid"]
    bestpath += [f",cnvresidoil"]
    bestpath += [f",cnvresidwater"]
    bestpath += [f",cnvresidgas"]
    # bestpath += [f",mbsminmaxresid"]
    bestpath += [f",iterationNumber"]
    bestpath += ["\n"]

    #bestpath += ["TStep[d],Defaulta"]
    # for i in range(NEWTONMAXIT + 1):
    #     #bestpath += [f",iterati{i}"]
    #     bestress += [f",iterati{i}"]
    #bestpath += ["\n"]
    # bestpath = [""]
    # bestress += ["\n"]
    ppath = []
    press = []
    for n in range(no_steps):
        # bestpath.append(f"{n},{tsteps[n]:.2e},{DEFAULTADAPTIVE:.2e},")
        # bestress.append(f"{tsteps[n]:.2e},{DEFAULTADAPTIVE:.2e},")
        ppath.append([])
        press.append([])
        bestvalue = mt.inf
        for k in range(NEWTONMAXIT+1):
            if k == 0:
                # bestpath += [f"{n}"]
                # bestpath += [","]
                # bestpath += [f"{DEFAULTADAPTIVE:.2e}"]
                # bestress += [f"{maxress[-1][n][k]:.2e}"]
                # bestpath += [f"{n}"]
                # bestpath += [","]
                bestpath += [f"{tsteps[n]:.2e}"]
                bestpath += [","]
                bestpath += [f"{DEFAULTADAPTIVE:.2e}"]
                bestpath += [","]
                # Forcing the first iteration tolerance to be the default value
                # bestpath += [f"{bestvalue:.2e}"]
                bestpath += [f"{DEFAULTADAPTIVE:.2e}"]
                bestpath += [","]
                bestpath += [f"{maxress[-1][n][k]:.2e}"]
                bestpath += [","]
                bestpath += [f"{rescnvo[-1][n][k]:.2e}"]
                bestpath += [","]
                bestpath += [f"{rescnvw[-1][n][k]:.2e}"]
                bestpath += [","]
                bestpath += [f"{rescnvg[-1][n][k]:.2e}"]
                bestpath += [","]
                bestpath += [f"{k}"]
                bestpath += "\n"
                ppath[-1].append(bestvalue)
                # press[-1].append(maxress[-1][n][k])
                # bestpath += [","]
                # bestress += [","]
            else:
                minress = mt.inf
                for i, value in enumerate(VALUES):
                    if k < len(maxress[i][n]):

                        valrescnvo = rescnvo[i][n][k]
                        valrescnvw = rescnvw[i][n][k]
                        valrescnvg = rescnvg[i][n][k]
                        # remove the value <= bestvalue to allow free mvmt of tols
                        # if maxress[i][n][k] < minress and value <= bestvalue:  
                        if maxress[i][n][k] < minress :
                            minress = maxress[i][n][k]
                            bestvalue = value
                if minress == mt.inf:
                    del bestpath[-1]
                    # del bestress[-1]
                    break
                # bestpath += [f"{n}"]
                # bestpath += [","]
                bestpath += [f"{tsteps[n]:.2e}"]
                bestpath += [","]
                bestpath += [f"{DEFAULTADAPTIVE:.2e}"]
                bestpath += [","]
                bestpath += [f"{bestvalue:.2e}"]
                bestpath += [","]
                bestpath += [f"{minress:.2e}"]
                bestpath += [","]
                bestpath += [f"{valrescnvo:.2e}"]
                bestpath += [","]
                bestpath += [f"{valrescnvw:.2e}"]
                bestpath += [","]
                bestpath += [f"{valrescnvg:.2e}"]
                bestpath += [","]
                bestpath += [f"{k}"]
                bestpath += "\n"

                ppath[-1].append(bestvalue)
                # press[-1].append(minress)
                if minress < 1:
                    break
                # bestpath += [","]
                # bestress += [","]
        if k == NEWTONMAXIT:
            del bestpath[-1]
            # del bestress[-1]
        bestpath += "\n"
        # bestress += "\n"

# WRITE
with open(
    f"{OUTFOL}/bestpath.csv",
    "w",
    encoding="utf8",
) as file:
    file.write("".join(bestpath))
with open(
    f"{OUTFOL}/bestress.csv",
    "w",
    encoding="utf8",
) as file:
    file.write("".join(bestress))
# PLOT
if len(VALUES) < 100:
    figs, axis = [], []
    for n in range(no_steps):
        fig, ax = plt.subplots()
        figs.append(fig)
        axis.append(ax)
    for i, val in enumerate(VALUES):
        for n in range(no_steps):
            axis[n].plot(
                iters[i][n],
                maxress[i][n],
                color=COLORS[i],
                label=VARIABLE + f"{val}",
                ls=LINESTYLE[i],
                lw=1,
            )
    for n in range(no_steps):
        axis[n].plot(
            range(len(press[n])),
            press[n],
            color=COLORS[-1],
            label="best path",
            ls="dotted",
            lw=1,
        )
    if os.path.exists(f"{OUTFOL}/adaptive/{CASE}.INFOITER"):
        for n in range(no_steps):
            axis[n].plot(
                iters[-1][n],
                maxress[-1][n],
                color="b",
                label="adaptive",
                ls="",
                marker="*",
                lw=3,
            )
    for n in range(no_steps):
        axis[n].set_ylabel("Max normalize residuals (cnvs and mbs) [-]")
        axis[n].set_yscale("log")
        axis[n].set_xlabel("Iteration no.")
        axis[n].set_title(
            CASE
            + f", report step {n} out of {info_ite[0][-1][info_itenam.index('ReportStep')]}"
        )
        axis[n].legend(prop={"size": 12})
        axis[n].xaxis.set_major_locator(MaxNLocator(integer=True))
        figs[n].savefig(f"{OUTFOL}/reportstep_{n}.png", bbox_inches="tight")
fig, ax = plt.subplots()
ax.set_title(CASE + f", Total no. report steps {int(info_ite[0][-1][info_itenam.index('ReportStep')])+1}")
ax.set_xscale("log")
ax.set_xlabel(VARIABLE[2:-1])
ax.set_ylabel(r"$\beta$I$_{N}$+$\alpha$I$_{L}$ ("+ r"$\beta$=" + f"{BETA}, "+ r"$\alpha$=" + f"{ALPHA})")
ax.plot(
    VALUES,
    BETA*np.array(newtit)+ALPHA*np.array(linit),
    color="k",
    marker="*",
    lw=1,
)
if os.path.exists(f"{OUTFOL}/adaptive/{CASE}.INFOITER"):
        ax.axhline(
        y = BETA*newtita+ALPHA*linita,
        color="b",
        lw=2,
    )
fig.savefig(f"{OUTFOL}/totaliterationswighted.png", bbox_inches="tight")
fig, ax = plt.subplots()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(CASE + f", Total no. report steps {int(info_ite[0][-1][info_itenam.index('ReportStep')])+1}")
ax.set_xscale("log")
ax.set_xlabel(VARIABLE[2:-1])
ax.set_ylabel(r"I$_{N}$")
ax.plot(
    VALUES,
    newtit,
    color="k",
    marker="*",
    lw=1,
)
if os.path.exists(f"{OUTFOL}/adaptive/{CASE}.INFOITER"):
        ax.axhline(
        y = newtita,
        color="b",
        lw=2,
    )
fig.savefig(f"{OUTFOL}/newtoniterations.png", bbox_inches="tight")
fig, ax = plt.subplots()
ax.set_title(CASE + f", Total no. report steps {int(info_ite[0][-1][info_itenam.index('ReportStep')])+1}")
ax.set_xscale("log")
ax.set_xlabel(VARIABLE[2:-1])
ax.set_ylabel(r"I$_{L}$")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.plot(
    VALUES,
    linit,
    color="k",
    marker="*",
    lw=1,
)
if os.path.exists(f"{OUTFOL}/adaptive/{CASE}.INFOITER"):
        ax.axhline(
        y = linita,
        color="b",
        lw=2,
    )
fig.savefig(f"{OUTFOL}/lineariterations.png", bbox_inches="tight")
