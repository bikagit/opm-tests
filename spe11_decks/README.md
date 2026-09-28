# SPE11 deck catalogue

Source: [OPM/pyopmspe11](https://github.com/OPM/pyopmspe11), commit `47f2b44fd6b8bebe7ac8392a1e92ae3b4f885bf7`.

## Which are the official benchmark configurations?

The 13 configurations under `benchmark/` are the current upstream OPM benchmark reproduction configurations: SPE11A r1–r5, SPE11B r1–r4, and SPE11C r1–r4. There is no single universal official OPM deck per case. The upstream benchmark gallery identifies this family as reproducing the OPM team results. Examples, regression tests, and locally derived variants are separately labelled below; they are not labelled as benchmark submissions.

Sources: [benchmark overview](https://opm.github.io/pyopmspe11/benchmark.html), [SPE11A](https://opm.github.io/pyopmspe11/benchmark/spe11a.html), [SPE11B](https://opm.github.io/pyopmspe11/benchmark/spe11b.html), [SPE11C](https://opm.github.io/pyopmspe11/benchmark/spe11c.html).

### SPE11C historical correction

Upstream documents that Well 1 in the submitted SPE11C results was approximately 100 m above the intended 300 m benchmark depth. Current files are current upstream reproduction inputs, not a guarantee of byte-for-byte historical submission inputs. The additional r5 comparison is locally derived from current r1 by setting dispersion to zero, exactly as described in upstream documentation; it is not a separately shipped configuration or an original submission.

## Available decks

| Category | Case / variant | Model | Grid dimensions | Cells | Deck | Configuration | Origin |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| benchmark | spe11a / r1_Cart_1cm | isothermal | 280 × 1 × 120 | 33,600 | [R1_CART_1CM.DATA](benchmark/spe11a/r1_Cart_1cm/R1_CART_1CM.DATA) | [TOML](configs/benchmark/spe11a/r1_Cart_1cm.toml) | generated |
| benchmark | spe11a / r2_Cart_1cm_capmax2500Pa | isothermal | 280 × 1 × 120 | 33,600 | [R2_CART_1CM_CAPMAX2500PA.DATA](benchmark/spe11a/r2_Cart_1cm_capmax2500Pa/R2_CART_1CM_CAPMAX2500PA.DATA) | [TOML](configs/benchmark/spe11a/r2_Cart_1cm_capmax2500Pa.toml) | upstream reference |
| benchmark | spe11a / r3_cp_1cmish_capmax2500Pa | isothermal | 280 × 1 × 183 | 51,240 | [R3_CP_1CMISH_CAPMAX2500PA.DATA](benchmark/spe11a/r3_cp_1cmish_capmax2500Pa/R3_CP_1CMISH_CAPMAX2500PA.DATA) | [TOML](configs/benchmark/spe11a/r3_cp_1cmish_capmax2500Pa.toml) | upstream reference |
| benchmark | spe11a / r4_Cart_1mm_capmax2500Pa | isothermal | 2800 × 1 × 1200 | 3,360,000 | [R4_CART_1MM_CAPMAX2500PA.DATA](benchmark/spe11a/r4_Cart_1mm_capmax2500Pa/R4_CART_1MM_CAPMAX2500PA.DATA) | [TOML](configs/benchmark/spe11a/r4_Cart_1mm_capmax2500Pa.toml) | generated |
| benchmark | spe11a / r5_Cart_1mm_capmax2500Pa_strictol | isothermal | 2800 × 1 × 1200 | 3,360,000 | [R5_CART_1MM_CAPMAX2500PA_STRICTOL.DATA](benchmark/spe11a/r5_Cart_1mm_capmax2500Pa_strictol/R5_CART_1MM_CAPMAX2500PA_STRICTOL.DATA) | [TOML](configs/benchmark/spe11a/r5_Cart_1mm_capmax2500Pa_strictol.toml) | generated |
| benchmark | spe11b / r1_Cart_10m | complete | 842 × 1 × 120 | 101,040 | [R1_CART_10M.DATA](benchmark/spe11b/r1_Cart_10m/R1_CART_10M.DATA) | [TOML](configs/benchmark/spe11b/r1_Cart_10m.toml) | upstream reference |
| benchmark | spe11b / r2_cp_10mish | complete | 842 × 1 × 157 | 132,194 | [R2_CP_10MISH.DATA](benchmark/spe11b/r2_cp_10mish/R2_CP_10MISH.DATA) | [TOML](configs/benchmark/spe11b/r2_cp_10mish.toml) | upstream reference |
| benchmark | spe11b / r3_cp_10mish_convective | convective | 842 × 1 × 157 | 132,194 | [R3_CP_10MISH_CONVECTIVE.DATA](benchmark/spe11b/r3_cp_10mish_convective/R3_CP_10MISH_CONVECTIVE.DATA) | [TOML](configs/benchmark/spe11b/r3_cp_10mish_convective.toml) | generated |
| benchmark | spe11b / r4_Cart_1m | complete | 8400 × 1 × 1200 | 10,080,000 | [R4_CART_1M.DATA](benchmark/spe11b/r4_Cart_1m/R4_CART_1M.DATA) | [TOML](configs/benchmark/spe11b/r4_Cart_1m.toml) | generated |
| benchmark | spe11c / r1_Cart_50m-50m-10m | complete | 170 × 102 × 120 | 2,080,800 | [R1_CART_50M-50M-10M.DATA](benchmark/spe11c/r1_Cart_50m-50m-10m/R1_CART_50M-50M-10M.DATA) | [TOML](configs/benchmark/spe11c/r1_Cart_50m-50m-10m.toml) | upstream reference |
| benchmark | spe11c / r2_cp_50m-50m-8mish | complete | 170 × 102 × 157 | 2,722,380 | [R2_CP_50M-50M-8MISH.DATA](benchmark/spe11c/r2_cp_50m-50m-8mish/R2_CP_50M-50M-8MISH.DATA) | [TOML](configs/benchmark/spe11c/r2_cp_50m-50m-8mish.toml) | upstream reference |
| benchmark | spe11c / r3_cp_50m-50m-8mish_convective | convective | 170 × 102 × 157 | 2,722,380 | [R3_CP_50M-50M-8MISH_CONVECTIVE.DATA](benchmark/spe11c/r3_cp_50m-50m-8mish_convective/R3_CP_50M-50M-8MISH_CONVECTIVE.DATA) | [TOML](configs/benchmark/spe11c/r3_cp_50m-50m-8mish_convective.toml) | generated |
| benchmark | spe11c / r4_cp_8m-8mish-8mish | complete | 1052 × 607 × 157 | 100,254,548 | [R4_CP_8M-8MISH-8MISH.DATA](benchmark/spe11c/r4_cp_8m-8mish-8mish/R4_CP_8M-8MISH-8MISH.DATA) | [TOML](configs/benchmark/spe11c/r4_cp_8m-8mish-8mish.toml) | generated |
| convergence | spe11b / full_cp0-z40mish-x40m | complete | 212 × 1 × 25 | 5,300 | [FULL_CP0-Z40MISH-X40M.DATA](convergence/full_cp0-z40mish-x40m/FULL_CP0-Z40MISH-X40M.DATA) | [TOML](configs/convergence/full_cp0-z40mish-x40m.toml) | generated from upstream template |
| convergence | spe11b / full_cp1-z20mish-x20m | complete | 422 × 1 × 55 | 23,210 | [FULL_CP1-Z20MISH-X20M.DATA](convergence/full_cp1-z20mish-x20m/FULL_CP1-Z20MISH-X20M.DATA) | [TOML](configs/convergence/full_cp1-z20mish-x20m.toml) | generated from upstream template |
| convergence | spe11b / full_cp2-z10mish-x10m | complete | 842 × 1 × 112 | 94,304 | [FULL_CP2-Z10MISH-X10M.DATA](convergence/full_cp2-z10mish-x10m/FULL_CP2-Z10MISH-X10M.DATA) | [TOML](configs/convergence/full_cp2-z10mish-x10m.toml) | generated from upstream template |
| convergence | spe11b / full_cp3-z5mish-x5m | complete | 1682 × 1 × 224 | 376,768 | [FULL_CP3-Z5MISH-X5M.DATA](convergence/full_cp3-z5mish-x5m/FULL_CP3-Z5MISH-X5M.DATA) | [TOML](configs/convergence/full_cp3-z5mish-x5m.toml) | generated from upstream template |
| convergence | spe11b / lower_cp0-z320mish-x320m | complete | — | — | Generation failed; [log](convergence/lower_cp0-z320mish-x320m/generation.log) | [TOML](configs/convergence/lower_cp0-z320mish-x320m.toml) | No runnable deck |
| convergence | spe11b / lower_cp1-z160mish-x160m | complete | 54 × 1 × 3 | 162 | [LOWER_CP1-Z160MISH-X160M.DATA](convergence/lower_cp1-z160mish-x160m/LOWER_CP1-Z160MISH-X160M.DATA) | [TOML](configs/convergence/lower_cp1-z160mish-x160m.toml) | generated from upstream template |
| convergence | spe11b / lower_cp2-z80mish-x80m | complete | 107 × 1 × 5 | 535 | [LOWER_CP2-Z80MISH-X80M.DATA](convergence/lower_cp2-z80mish-x80m/LOWER_CP2-Z80MISH-X80M.DATA) | [TOML](configs/convergence/lower_cp2-z80mish-x80m.toml) | generated from upstream template |
| convergence | spe11b / lower_cp3-z40mish-x40m | complete | 212 × 1 × 9 | 1,908 | [LOWER_CP3-Z40MISH-X40M.DATA](convergence/lower_cp3-z40mish-x40m/LOWER_CP3-Z40MISH-X40M.DATA) | [TOML](configs/convergence/lower_cp3-z40mish-x40m.toml) | generated from upstream template |
| convergence | spe11b / lower_cp4-z20mish-x20m | complete | 422 × 1 × 19 | 8,018 | [LOWER_CP4-Z20MISH-X20M.DATA](convergence/lower_cp4-z20mish-x20m/LOWER_CP4-Z20MISH-X20M.DATA) | [TOML](configs/convergence/lower_cp4-z20mish-x20m.toml) | generated from upstream template |
| convergence | spe11b / lower_cp5-z10mish-x10m | complete | 842 × 1 × 37 | 31,154 | [LOWER_CP5-Z10MISH-X10M.DATA](convergence/lower_cp5-z10mish-x10m/LOWER_CP5-Z10MISH-X10M.DATA) | [TOML](configs/convergence/lower_cp5-z10mish-x10m.toml) | generated from upstream template |
| convergence | spe11b / lower_cp6-z5mish-x5m | complete | 1682 × 1 × 73 | 122,786 | [LOWER_CP6-Z5MISH-X5M.DATA](convergence/lower_cp6-z5mish-x5m/LOWER_CP6-Z5MISH-X5M.DATA) | [TOML](configs/convergence/lower_cp6-z5mish-x5m.toml) | generated from upstream template |
| derived | spe11c / r5_Cart_50m-50m-10m_no_dispersion | complete | 170 × 102 × 120 | 2,080,800 | [R5_CART_50M-50M-10M_NO_DISPERSION.DATA](derived/spe11c/r5_Cart_50m-50m-10m_no_dispersion/R5_CART_50M-50M-10M_NO_DISPERSION.DATA) | [TOML](configs/derived/spe11c/r5_Cart_50m-50m-10m_no_dispersion.toml) | generated |
| examples | spe11a / spe11a | isothermal | 55 × 1 × 74 | 4,070 | [SPE11A.DATA](examples/spe11a/SPE11A.DATA) | [TOML](configs/examples/spe11a.toml) | generated |
| examples | spe11b / spe11b | complete | 83 × 1 × 58 | 4,814 | [SPE11B.DATA](examples/spe11b/SPE11B.DATA) | [TOML](configs/examples/spe11b.toml) | generated |
| examples | spe11b / spe11b_convective | convective | 83 × 1 × 58 | 4,814 | [SPE11B_CONVECTIVE.DATA](examples/spe11b_convective/SPE11B_CONVECTIVE.DATA) | [TOML](configs/examples/spe11b_convective.toml) | generated |
| examples | spe11c / spe11c | complete | 44 × 27 × 12 | 14,256 | [SPE11C.DATA](examples/spe11c/SPE11C.DATA) | [TOML](configs/examples/spe11c.toml) | generated |
| examples | spe11b / special_issue_convective | convective | — | — | Generation failed; [log](examples/special_issue_convective/generation.log) | [TOML](configs/examples/special_issue_convective.toml) | No runnable deck |
| tests | spe11b / input | complete | 72 × 1 × 52 | 3,744 | [INPUT.DATA](tests/input/INPUT.DATA) | [TOML](configs/tests/input.toml) | generated |
| tests | spe11a / spe11a | isothermal | 55 × 1 × 74 | 4,070 | [SPE11A.DATA](tests/spe11a/SPE11A.DATA) | [TOML](configs/tests/spe11a.toml) | generated |
| tests | spe11a / spe11a_cartesian | isothermal | 28 × 1 × 12 | 336 | [SPE11A_CARTESIAN.DATA](tests/spe11a_cartesian/SPE11A_CARTESIAN.DATA) | [TOML](configs/tests/spe11a_cartesian.toml) | generated |
| tests | spe11a / spe11a_corner-point | isothermal | 28 × 1 × 11 | 308 | [SPE11A_CORNER-POINT.DATA](tests/spe11a_corner-point/SPE11A_CORNER-POINT.DATA) | [TOML](configs/tests/spe11a_corner-point.toml) | generated |
| tests | spe11a / spe11a_data_format | isothermal | 28 × 1 × 12 | 336 | [SPE11A_DATA_FORMAT.DATA](tests/spe11a_data_format/SPE11A_DATA_FORMAT.DATA) | [TOML](configs/tests/spe11a_data_format.toml) | generated |
| tests | spe11b / spe11b_cartesian | complete | 86 × 1 × 12 | 1,032 | [SPE11B_CARTESIAN.DATA](tests/spe11b_cartesian/SPE11B_CARTESIAN.DATA) | [TOML](configs/tests/spe11b_cartesian.toml) | generated |
| tests | spe11b / spe11b_corner-point | complete | 84 × 1 × 11 | 924 | [SPE11B_CORNER-POINT.DATA](tests/spe11b_corner-point/SPE11B_CORNER-POINT.DATA) | [TOML](configs/tests/spe11b_corner-point.toml) | generated |
| tests | spe11b / spe11b_data_format | complete | 86 × 1 × 12 | 1,032 | [SPE11B_DATA_FORMAT.DATA](tests/spe11b_data_format/SPE11B_DATA_FORMAT.DATA) | [TOML](configs/tests/spe11b_data_format.toml) | generated |
| tests | spe11c / spe11c | complete | 135 × 11 × 18 | 26,730 | [SPE11C.DATA](tests/spe11c/SPE11C.DATA) | [TOML](configs/tests/spe11c.toml) | generated |
| tests | spe11c / spe11c_cartesian | complete | 18 × 12 × 12 | 2,592 | [SPE11C_CARTESIAN.DATA](tests/spe11c_cartesian/SPE11C_CARTESIAN.DATA) | [TOML](configs/tests/spe11c_cartesian.toml) | generated |
| tests | spe11c / spe11c_corner-point | complete | 18 × 12 × 11 | 2,376 | [SPE11C_CORNER-POINT.DATA](tests/spe11c_corner-point/SPE11C_CORNER-POINT.DATA) | [TOML](configs/tests/spe11c_corner-point.toml) | generated |
| tests | spe11c / spe11c_data_format | complete | 44 × 27 × 18 | 21,384 | [SPE11C_DATA_FORMAT.DATA](tests/spe11c_data_format/SPE11C_DATA_FORMAT.DATA) | [TOML](configs/tests/spe11c_data_format.toml) | generated |

## Scope and usage

This catalogue materializes every shipped TOML case under upstream `benchmark/`, `examples/`, and `tests/configs/`, plus the documented derived SPE11C r5. Arbitrary resolutions and physics settings admit unlimited additional decks. The 11 documented convergence-study configurations (four full-domain and seven lower-domain) are also included from the upstream Mako template and labelled separately. Further optional refinements are parameterized in `pyopmspe11/convergence/convergence.py` and are not enumerated here.

All decks retain their companion include files. The six supplied reference variants are copied directly (the zipped C grid is extracted); other variants are generated with the installed upstream generator in deck-only mode. No simulations were run. All listed include references were checked.

Run a deck from its containing directory:

```bash
flow CASE.DATA --output-dir=output
```

For benchmark reproduction, also use the `flow` command and solver options in the corresponding TOML. In particular, SPE11A r5 differs from r4 in solver tolerances outside the DATA file. The benchmark documentation linked above supplies reporting and postprocessing commands. Plain `flow CASE.DATA` does not reproduce all those solver/reporting settings.

The original convenient `SPE11A/`, `SPE11B/`, and `SPE11C/` folders remain: they correspond to A r2, B r1, and C r1.

Regenerate the fixed collection with `.venv/bin/python generate_all.py` and the documented convergence series with `.venv/bin/python generate_convergence.py`; rebuild and verify this catalogue with `.venv/bin/python build_catalog.py`. The isolated environment contains the generator and dependencies. The upstream README specifies Flow 2026.04 or current master; master-specific configuration features still need a compatible executable.

## Unavailable upstream configurations

The upstream lower-domain 320 m convergence template produces z_n[15] = z_n[16] = 0; the current generator rejects these values. Its unmodified TOML and error log are retained. No runnable deck is claimed for that configuration.

The special_issue_convective example also fails in the current upstream generator with a corner-point refinement array-shape mismatch (50 values into 49 slots). Its unchanged configuration and traceback are retained.

Incomplete generations: convergence/lower_cp0-z320mish-x320m, examples/special_issue_convective