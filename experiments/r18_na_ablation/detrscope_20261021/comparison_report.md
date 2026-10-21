# detrscope cross-model comparison

Images: 8 · GT: 871 · eligible GT: 871

Native/observed/restored output verification: baseline=exact_native_observed_restored; candidate=exact_condition_observation_identity_intervention_and_restoration

Pairs use image and GT identity. Query ids remain model-local. Candidate coverage and Hungarian responsibility are diagnostic and do not define TP/FP or AP. Probability margins may reflect different calibration. Observations do not establish causes.

Lost/Gained only from supplied CocoEvaluator annotation/protocol-identical matches. Annotation/protocol, image scope and raw GT identities are checked. Prediction-to-recorded-model binding is not verified; official Lost/Gained describe the supplied result pair, not proven outputs of the recorded models.

All changes are candidate minus baseline. Coverage has no confidence gate. Missing candidates are excluded from paired deltas.

| Group | Value | GT | Eligible | Geometry ΔIoU | Geometry Δmargin | Correct-class ΔIoU | Correct-class Δmargin | Baseline coverage@.50 | Candidate coverage@.50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | all | 871 | 871 | -0.002155 (n=871) | -0.014420 (n=871) | -0.000703 (n=851) | -0.009399 (n=851) | 0.640643 (n=871) | 0.639495 (n=871) |
| class | awning-tricycle | 34 | 34 | -0.002154 (n=34) | 0.016056 (n=34) | 0.016811 (n=20) | 0.000000 (n=20) | 0.264706 (n=34) | 0.294118 (n=34) |
| class | bicycle | 17 | 17 | 0.011420 (n=17) | 0.029512 (n=17) | 0.004183 (n=13) | 0.000000 (n=13) | 0.058824 (n=17) | 0.058824 (n=17) |
| class | car | 229 | 229 | 0.001113 (n=229) | -0.039153 (n=229) | 0.001885 (n=229) | -0.024503 (n=229) | 0.921397 (n=229) | 0.930131 (n=229) |
| class | motor | 254 | 254 | -0.002277 (n=254) | -0.017238 (n=254) | -0.001907 (n=254) | -0.010396 (n=254) | 0.688976 (n=254) | 0.685039 (n=254) |
| class | pedestrian | 154 | 154 | -0.003930 (n=154) | 0.002372 (n=154) | -0.006666 (n=154) | 0.003388 (n=154) | 0.564935 (n=154) | 0.551948 (n=154) |
| class | people | 154 | 154 | -0.002712 (n=154) | 0.002152 (n=154) | 0.000709 (n=154) | 0.000382 (n=154) | 0.448052 (n=154) | 0.441558 (n=154) |
| class | tricycle | 17 | 17 | -0.043297 (n=17) | -0.036919 (n=17) | -0.002646 (n=15) | 0.000000 (n=15) | 0.058824 (n=17) | 0.058824 (n=17) |
| class | truck | 1 | 1 | 0.049716 (n=1) | 0.370598 (n=1) | 0.000929 (n=1) | 0.000000 (n=1) | 0.000000 (n=1) | 0.000000 (n=1) |
| class | van | 11 | 11 | 0.003182 (n=11) | -0.063825 (n=11) | 0.001846 (n=11) | -0.029775 (n=11) | 0.454545 (n=11) | 0.454545 (n=11) |
| density | crowded | 823 | 823 | -0.002479 (n=823) | -0.014417 (n=823) | -0.001076 (n=803) | -0.007871 (n=803) | 0.631835 (n=823) | 0.631835 (n=823) |
| density | moderate | 48 | 48 | 0.003394 (n=48) | -0.014472 (n=48) | 0.005544 (n=48) | -0.034973 (n=48) | 0.791667 (n=48) | 0.770833 (n=48) |
| size | large | 34 | 34 | 0.000980 (n=34) | -0.038989 (n=34) | 0.011613 (n=34) | -0.023916 (n=34) | 0.852941 (n=34) | 0.882353 (n=34) |
| size | medium | 366 | 366 | -0.000815 (n=366) | -0.032994 (n=366) | 0.000283 (n=356) | -0.021621 (n=356) | 0.786885 (n=366) | 0.792350 (n=366) |
| size | small | 471 | 471 | -0.003423 (n=471) | 0.001788 (n=471) | -0.002372 (n=461) | 0.001109 (n=461) | 0.511677 (n=471) | 0.503185 (n=471) |

## Diagnostic GT state transitions

At IoU 0.50 with no confidence gate. These states do not define official TP/FN.

| Baseline state | Candidate state | Count | Eligible GT denominator |
| --- | --- | ---: | ---: |
| class_confusion | class_confusion | 108 | 871 |
| class_confusion | correct_class_coverage | 4 | 871 |
| class_confusion | insufficient_geometry | 7 | 871 |
| correct_class_coverage | class_confusion | 2 | 871 |
| correct_class_coverage | correct_class_coverage | 549 | 871 |
| correct_class_coverage | insufficient_geometry | 7 | 871 |
| insufficient_geometry | class_confusion | 4 | 871 |
| insufficient_geometry | correct_class_coverage | 4 | 871 |
| insufficient_geometry | insufficient_geometry | 186 | 871 |

## Supplied official result pair: same-GT transitions

Prediction-to-recorded-model binding is unverified.

| IoU | Lost | Gained | Stable TP | Stable FN |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | 7 | 9 | 539 | 316 |
| 0.55 | 11 | 12 | 511 | 337 |
| 0.6 | 10 | 10 | 472 | 379 |
| 0.65 | 15 | 13 | 426 | 417 |
| 0.7 | 13 | 10 | 376 | 472 |
| 0.75 | 23 | 17 | 300 | 531 |
| 0.8 | 13 | 18 | 228 | 612 |
| 0.85 | 16 | 12 | 161 | 682 |
| 0.8999999999999999 | 14 | 7 | 89 | 761 |
| 0.95 | 3 | 9 | 21 | 838 |

Full per-GT views: `/home/zjnu/文档/ZHJ_WorkSpace/ultralytics/runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/detrscope_20261021/zero_d3_comparison/per_gt.jsonl`（仅服务器） · `/home/zjnu/文档/ZHJ_WorkSpace/ultralytics/runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/detrscope_20261021/zero_d3_comparison/per_gt.csv`（仅服务器）. All class, size and density metrics: [CSV](comparison_group_stats.csv). Grouped diagnostic transitions: [CSV](comparison_diagnostic_transitions.csv).

Provenance, original record manifests and raw SHA256 identities: [manifest](comparison_manifest.json).

## Measurement definitions

- **causality**: Observations do not establish mechanisms or retraining benefits
- **correct_argmax**: Highest-IoU valid normal query whose argmax class equals the GT class; no confidence gate
- **coverage**: GT has a qualifying candidate with IoU>=0.50 or >=0.75; not official recall/AP
- **delta**: Candidate minus baseline; null when either selected candidate is unavailable
- **density**: Eligible GTs per image: sparse <=10, moderate 11..50, crowded >50
- **diagnostic_states**: At IoU>=0.50, no confidence gate: no_valid_candidate; insufficient_geometry (no geometrically qualifying query); class_confusion (geometry covered, no correct-argmax query covers GT); correct_class_coverage. These are not TP/FN labels
- **excluded**: Ignored/crowd/invalid GTs retained in rows but excluded from metric denominators
- **geometry**: Highest-IoU valid normal query regardless of predicted class; no confidence gate
- **margin**: p(GT)-max(p(other foreground classes)); single-class competitor=0; probability calibration may differ across models
- **official**: Lost/Gained only from supplied CocoEvaluator annotation/protocol-identical matches
- **pairing**: Same image and GT identity/metadata only; query ids are model-local and never paired across models
- **responsibility**: Shared diagnostic Hungarian assignment with cost=(1-IoU)+0.25*(1-pGT), no confidence/IoU gate; not TP/FP
- **size**: Original-pixel area: small [0,1024), medium [1024,9216), large [9216,infinity); diagnostic strata
- **stage**: Final executed native decoder stage; decoder depths may differ
