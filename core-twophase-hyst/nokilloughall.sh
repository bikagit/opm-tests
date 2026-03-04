#!/bin/bash
set -e  # Exit on error
set -u  # Treat unset variables as error

# === CONFIGURATION ===
# === CONFIGURATION ===
CASE="/CASE-B/test020"
FOLDER="/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1"${CASE}
#FOLDER="/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/CASE-B/test006"
MODELFOLDER="/Users/macbookn/activopmwkspc/edgedev/opm-tests/ml-simulations/espen_hyst_model/boilerhyst"${CASE}
FLOW_BIN="/Users/macbookn/activopmwkspc/edgedev/build/opm-simulators/bin/flow"
MODEL_PATH="/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/mlhyst/models/oldmodelkrnw.model"
PLOPM_ENV="/Users/macbookn/activopmwkspc/pyDavid/plopm/vplopm/bin/activate"

# === FUNCTIONS ===

run_flow() {
    local output_dir=$1
    local data_file=$2
    local extra_args=${3:-""}

    echo "[INFO] Running flow simulation: $data_file -> $output_dir"
    mkdir -p "$output_dir"
    $FLOW_BIN \
        --newton-max-iterations=20 \
        --relaxed-max-pv-fraction=0 \
        --output-extra-convergence-info=steps,iterations \
        --enable-opm-rst-file=true \
        --newton-min-iterations=1 \
        --enable-well-operability-check=false \
        --enable-write-all-solutions=true \
        --min-time-step-before-shutting-problematic-wells-in-days=1e-99 \
        --output-dir="$output_dir" $extra_args "$data_file"
}

update_model() {
    local model_file=$1
    echo "[INFO] Updating ML model: $model_file"
    rm -rf "$MODEL_PATH"
    cp -r "$MODELFOLDER/model/$model_file" "$MODEL_PATH"
}

generate_plots() {
    local prefix=$1
    local output=$2
    local colors=$3
    local labels=$4
    local extra_inputs=$5
    local tunits=$6
    local save_prefix=$7
    local num_cells=$8
    local xlnums=$9
    # local size=$10

    echo "[INFO] Generating plots for $prefix"
    for cell in $(seq 1 $num_cells); do
        plopm -i "${extra_inputs}" --variable "swat" \
        -s "${cell},1,1 ${cell},1,1 ${cell},1,1" \
        -c "${colors}" -lw "1.5,1.5,1.5" -e "solid,dashed,dotted" \
        -labels "${labels}" -ylabel "Saturation (Sw) at cell ${cell}" \
        -yformat .2f -xlnum ${xlnums} -tunits ${tunits} -f 17 -save=${save_prefix}cell${cell} --output=${output}
    done
}

# === MAIN WORKFLOW ===

echo "[INFO] Starting simulations..."
mkdir -p "$FOLDER"/Figures
# Classic simulations
# # run_flow "$FOLDER/ClassicKillough-hystnew-output" "$FOLDER/CORE_ExampleKillough.DATA"
# run_flow "$FOLDER/ClassicCarlson-hystnew-output" "$FOLDER/CORE_ExampleCarlson.DATA"
# run_flow "$FOLDER/Nohysteresis-output" "$FOLDER/CORE_Examplenohyst.DATA"

# # ML-based simulations
# update_model "oldmodelkrnwCORE_ExampleKillough.model"
# run_flow "$FOLDER/Killoughmlhystnew-output" "$FOLDER/CORE_ExampleKillough.DATA" "--activate-m-l-rel-p-erm=True"

update_model "oldmodelkrnwCORE_ExampleCarlson.model"
run_flow "$FOLDER/Carlsonmlhystnew-output" "$FOLDER/CORE_ExampleCarlson.DATA" "--activate-m-l-rel-p-erm=True"

update_model "oldmodelkrnwCORE_Example.model"
run_flow "$FOLDER/LSLBMmlhystnew-output" "$FOLDER/CORE_Example.DATA" "--activate-m-l-rel-p-erm=True"

# Activate plotting environment
source "$PLOPM_ENV"

echo "[INFO] Generating plots..."
# generate_plots Killough $FOLDER/Figures/KilloughFiguresOutput "g,y,k" "ML-Killough-hysteresis  Killough hysteresis  No-hysteresis" \
# "$FOLDER/Killoughmlhystnew-output/CORE_EXAMPLEKILLOUGH $FOLDER/ClassicKillough-hystnew-output/CORE_EXAMPLEKILLOUGH $FOLDER/Nohysteresis-output/CORE_EXAMPLENOHYST" h curvekillough 9

generate_plots Carlson $FOLDER/Figures/CarlsonFiguresOutput "b,c,k" "ML-Carlson-hyst  Carlson hyst  No-hyst" \
"$FOLDER/Carlsonmlhystnew-output/CORE_EXAMPLECARLSON $FOLDER/ClassicCarlson-hystnew-output/CORE_EXAMPLECARLSON $FOLDER/Nohysteresis-output/CORE_EXAMPLENOHYST" d curvecarlson 9 10

generate_plots LSLBM $FOLDER/Figures/LSLBMFiguresOutput "b,g,r" "ML-Carlson-hyst  ML-LS-LBM" \
"$FOLDER/Carlsonmlhystnew-output/CORE_EXAMPLECARLSON $FOLDER/LSLBMmlhystnew-output/CORE_EXAMPLE" d curvelslbm 9 10

# Combined plot
for cell in $(seq 1 9); do
    plopm -i "$FOLDER/Carlsonmlhystnew-output/CORE_EXAMPLECARLSON $FOLDER/LSLBMmlhystnew-output/CORE_EXAMPLE $FOLDER/Nohysteresis-output/CORE_EXAMPLENOHYST" \
    --variable "swat" \
    -s "${cell},1,1 ${cell},1,1 ${cell},1,1" \
    -c "b,r,k" -lw "1.5,1.5,1.5" -e "solid,solid,dotted" \
    -labels "ML-Carlson-hysteresis  ML-LS-LBM  No-hysteresis" \
    -ylabel "Saturation (Sw) at cell ${cell}" -yformat .2f -xlnum 10 --size=17 -tunits d -save=lslbmcell${cell} --output=$FOLDER/Figures/LSLBMOutputFiguresfull
done



deactivate



# cells=(1 2 3 4 5 6 7 8)

# for cell in "${cells[@]}"; do

#     # python3 compare_unrst.py \
#     #     "$FOLDER/ClassicKillough-hystnew-output/CORE_EXAMPLEKILLOUGH.UNRST" \
#     #     "$FOLDER/ClassicKillough-hystnew-output/CORE_EXAMPLEKILLOUGH.UNRST" \
#     #     --legends "Killough" "Killough" \
#     #     --cells "$cell" 1 1 \
#     #     --outdir "$FOLDER/Figures/my_plotskillough"

    # Killough comparison
    # python3 compare_unrst.py \
    #     "$FOLDER/Nohysteresis-output/CORE_EXAMPLENOHYST.UNRST" \
    #     "$FOLDER/ClassicKillough-hystnew-output/CORE_EXAMPLEKILLOUGH.UNRST" \
    #     "$FOLDER/Killoughmlhystnew-output/CORE_EXAMPLEKILLOUGH.UNRST" \
    #     --csv-names $MODELFOLDER/runNohystKillough.csv $MODELFOLDER/runKillough.csv $MODELFOLDER/runMLkillough.csv \
    #     --legends "Nohyst" "Killough" "MLKillough" \
    #     --outdir "$FOLDER/Figures/my_plotskillough"\
    #     --model-colors black olive green
        # --dates "11.jan 2015" '12.JAN 2015' '13.JAN 2015' '14.JAN 2015'
        # --dates "01.jan 2015" '02.JAN 2015' '03.JAN 2015' '04.JAN 2015'

    # Carlson comparison
    # python3 compare_unrst.py \
    #     "$FOLDER/Nohysteresis-output/CORE_EXAMPLENOHYST.UNRST" \
    #     "$FOLDER/ClassicCarlson-hystnew-output/CORE_EXAMPLECARLSON.UNRST" \
    #     "$FOLDER/Carlsonmlhystnew-output/CORE_EXAMPLECARLSON.UNRST" \
    #     --csv-names $MODELFOLDER/runNohystCarlson.csv $MODELFOLDER/runCarlson.csv $MODELFOLDER/runMLCarlson.csv \
    #     --legends "Nohyst" "Carlson" "MLCarlson" \
    #     --outdir "$FOLDER/Figures/my_plotcarlson" \
    #     --model-colors black cyan blue \
    #     --dates "05.jan 2015" '06.JAN 2015' '07.JAN 2015'        # --cells "$cell" 1 1 
    # # LSLBM comparison
    # python3 compare_unrst.py \
    #     "$FOLDER/Nohysteresis-output/CORE_EXAMPLENOHYST.UNRST" \
    #     "$FOLDER/Carlsonmlhystnew-output/CORE_EXAMPLECARLSON.UNRST" \
    #     "$FOLDER/LSLBMmlhystnew-output/CORE_EXAMPLE.UNRST" \
    #     --csv-names $MODELFOLDER/runNohystLS.csv $MODELFOLDER/runLSMLCARLSON.csv $MODELFOLDER/runMLLSLBM.csv \
    #     --legends "Nohyst" "Carlson" "LSLBM" \
    #     --outdir "$FOLDER/Figures/my_plotslslbm"\
    #     --model-colors black  blue orange \
    #     --dates "05.jan 2015" '06.JAN 2015' '07.JAN 2015'        # --cells "$cell" 1 1 

# done

echo "[INFO] ALL NUMERICAL TESTS COMPLETED"
