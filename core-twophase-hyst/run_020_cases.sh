#!/bin/bash
# Run ML-Carlson and ML-LS-LBM (no Killough) Flow simulations and plots
# for both CASE-A/test020 and CASE-B/test020.
set -euo pipefail

BASEDIR="/Users/macbookn/activopmwkspc/edgedev/opm-tests/core-twophase-hyst"
MLBASE="/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst"
FLOW_BIN="/Users/macbookn/activopmwkspc/edgedev/build/opm-simulators/bin/flow"
MODEL_SLOT="$BASEDIR/mlhyst/models/oldmodelkrnw.model"
PLOPM_ENV="/Users/macbookn/activopmwkspc/pyDavid/plopm/vplopm/bin/activate"

run_flow() {
    local outdir=$1 datafile=$2 extra=${3:-""}
    echo "[INFO] flow: $datafile -> $outdir"
    mkdir -p "$outdir"
    $FLOW_BIN \
        --newton-max-iterations=20 --relaxed-max-pv-fraction=0 \
        --output-extra-convergence-info=steps,iterations \
        --enable-opm-rst-file=true --newton-min-iterations=1 \
        --enable-well-operability-check=false \
        --enable-write-all-solutions=true \
        --min-time-step-before-shutting-problematic-wells-in-days=1e-99 \
        --output-dir="$outdir" $extra "$datafile"
}

update_model() {
    local src=$1
    echo "[INFO] Deploying model: $(basename $src)"
    rm -f "$MODEL_SLOT"
    cp "$src" "$MODEL_SLOT"
}

run_case() {
    local CASE=$1   # e.g. CASE-A/test020
    local FOLDER="$BASEDIR/$CASE"
    local MFOLDER="$MLBASE/$CASE"
    echo ""
    echo "====== $CASE ======"
    mkdir -p "$FOLDER/Figures"

    # Classic Carlson + no-hysteresis
    run_flow "$FOLDER/ClassicCarlson-hystnew-output" "$FOLDER/CORE_ExampleCarlson.DATA"
    run_flow "$FOLDER/Nohysteresis-output"           "$FOLDER/CORE_Examplenohyst.DATA"

    # ML simulations (Carlson and LS-LBM only — Killough diverges at Swi=0.20)
    update_model "$MFOLDER/model/oldmodelkrnwCORE_ExampleCarlson.model"
    run_flow "$FOLDER/Carlsonmlhystnew-output"  "$FOLDER/CORE_ExampleCarlson.DATA" "--activate-m-l-rel-p-erm=True"

    update_model "$MFOLDER/model/oldmodelkrnwCORE_Example.model"
    run_flow "$FOLDER/LSLBMmlhystnew-output"    "$FOLDER/CORE_Example.DATA"        "--activate-m-l-rel-p-erm=True"

    # Plots
    source "$PLOPM_ENV"
    for cell in $(seq 1 9); do
        plopm -i "$FOLDER/Carlsonmlhystnew-output/CORE_EXAMPLECARLSON $FOLDER/ClassicCarlson-hystnew-output/CORE_EXAMPLECARLSON $FOLDER/Nohysteresis-output/CORE_EXAMPLENOHYST" \
            --variable swat -s "${cell},1,1 ${cell},1,1 ${cell},1,1" \
            -c "b,c,k" -lw "1.5,1.5,1.5" -e "solid,dashed,dotted" \
            -labels "ML-Carlson-hyst  Carlson hyst  No-hyst" \
            -ylabel "Saturation (Sw) at cell ${cell}" -yformat .2f -xlnum 10 -tunits d \
            -save=curvecarlsoncell${cell} --output="$FOLDER/Figures/CarlsonFiguresOutput"

        plopm -i "$FOLDER/Carlsonmlhystnew-output/CORE_EXAMPLECARLSON $FOLDER/LSLBMmlhystnew-output/CORE_EXAMPLE $FOLDER/Nohysteresis-output/CORE_EXAMPLENOHYST" \
            --variable swat -s "${cell},1,1 ${cell},1,1 ${cell},1,1" \
            -c "b,r,k" -lw "1.5,1.5,1.5" -e "solid,solid,dotted" \
            -labels "ML-Carlson-hyst  ML-LS-LBM  No-hyst" \
            -ylabel "Saturation (Sw) at cell ${cell}" -yformat .2f -xlnum 10 -tunits d \
            -save=lslbmcell${cell} --output="$FOLDER/Figures/LSLBMOutputFiguresfull"
    done
    deactivate
    echo "[INFO] $CASE done."
}

run_case "CASE-A/test020"
run_case "CASE-B/test020"
echo ""
echo "[INFO] ALL 020 CASES COMPLETE"
