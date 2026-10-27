# Ball localization and repeated SAM tracking

October 27, 2026. NBHL ball hockey only. Work ran on Modal H100 GPUs. The directory date follows the preceding ball experiments.

## What was tested

Nine new detector fine-tunes, two existing detector baselines, temporal input and synthetic-motion variants, hard-negative mining, sequence-level OpenAI relocalization, short SAM/TAP tracks, SAM prompt variants, and matched runtime controls. Model weights and raw footage stay on the Modal volume; this repository contains code, protocols, labels, hashes and numerical evidence.

The strongest established finding is a SAM runtime problem: asynchronous GPU-to-CPU state transfers can corrupt the starting mask. Keeping the small tracking state on the GPU fixes the measured failure. This is separate from whether the starting point is really the ball.

## Detector training and new-game test

The existing RGB heatmap model was expanded into RGB, three-frame RGB, and three-frame RGB plus signed motion-difference variants. The backbone is ResNet18 with a stride-two decoder. Training uses native 320-pixel crops, camera-aligned neighboring frames at offsets −2 and +2, 70% positive and 30% negative crops. Synthetic variants paste an actual small ball patch along a short motion path, with blur and occasional middle-frame occlusion. Later variants initialize only the center frame from the RGB checkpoint, or add hard negatives mined exclusively from previously reviewed training images.

There are 226 positive training frames: 92 earlier CA/GA frames, 23 earlier manual NJ frames, and 111 published detector/OpenAI pseudo-labels from the first 100 seconds of the current replay. Pseudo-labels are explicitly not human truth. Five overlapping NJ0900 validation frames were removed from selection before training; the amendment and original hashes are recorded. Eleven NJ2400 frames are the remaining development split. No new-game reference enters training.

Checkpoint selection uses development crop F1. Full-frame operating points use only the eleven development frames. Before opening new-game scores, `rgb-expanded` was chosen by development F1, false positives, then simplicity. Its development result did **not** transfer into a win over YOLO.

Two additional NBHL recordings, four fixed 90-second intervals, were reviewed at 120 predetermined times without detector overlays. Only 27 frames established a visible ball and one established absence; 92 unresolved frames are excluded, not treated as negatives. These are sparse assistant annotations, not independent human ground truth. See `fresh-reference.json`, `extra-reference.json`, their source image hashes, and the sealed selection/evaluation files.

| Detector | Correct within 25 px / 27 visible | Wrong location | Missing | False on the one absent frame |
|---|---:|---:|---:|---:|
| Existing YOLO | 22 | 4 | 1 | 0 |
| Existing RGB heatmap | 18 | 2 | 7 | 0 |
| Expanded RGB | 21 | 6 | 0 | 1 |
| Three-frame RGB | 16 | 11 | 0 | 0 |
| Motion input | 17 | 9 | 1 | 0 |
| Motion + synthetic | 15 | 11 | 1 | 0 |
| Center-initialized RGB temporal | 19 | 3 | 5 | 0 |
| Center-initialized motion | 17 | 2 | 8 | 0 |
| Center motion + synthetic | 20 | 6 | 1 | 0 |
| RGB + hard negatives | 21 | 6 | 0 | 0 |
| Center temporal + hard negatives | 19 | 7 | 1 | 0 |

A wrong location contributes both a false positive and a false negative. Unknown-frame outputs are counted separately. These numbers do not measure continuous tracking, possession, shots, or full-game precision. The detector operating points and model choice were frozen before these scores. Results do not justify replacing the existing detector with a temporal CNN.

A final check used two further disjoint intervals at 30:00–31:30 in the same recordings. Thirty fixed raw frames were reviewed before opening predictions: six visible and 24 unresolved. The unchanged YOLO policy found all six within 12 pixels. Motion plus synthetic data also found all six, but did worse on the first 120 samples. Across all 150 fixed samples, YOLO gives **28 correct, four wrong and one missing on 33 visible frames**; 116 unresolved frames remain excluded. This is additional time coverage, not two more unseen game identities. `results/all-intervals.json` combines the disjoint samples.

## False positives hidden by unresolved references

**The visible-frame result does not establish precision.** A further audit selected 60 evenly spaced outputs from the 88 originally unresolved frames on which frozen YOLO produced a point. Chronological close-ups confirmed **52 wrong objects, three visible balls and five still unresolved**. Typical mistakes are scoreboard text, board bolts, red shoes, helmets and markings. The original 150 reference labels are unchanged; the audit is separate, post-prediction assistant evidence. Wrong object means the output is wrong, not that the ball is absent elsewhere in the frame.

The new replay-trained rejection gate removes 46 of those 52 wrong outputs while keeping two of the three confirmed balls. It also reduces the original visible-frame correct count from 28 to 22 of 33. That is a useful false-positive/recall tradeoff, not a general model win. The earliest temporal-head gate rejects all 52 wrong objects but retains just one of the three balls and only 14 of the original 33 visible cases. All verifier predictions were complete before this audit. `transfer-output-audit.json` and `results/transfer-output-audit-gates.json` preserve every verdict and acceptance decision.

The final surface-context experiment uses pretrained SAM3 with the text prompt `playing surface`, unions masks at confidence 0.3, forms a convex hull, and tests margins of zero and 2.5% of image height. No ball reference or hand-drawn rink polygon enters mask generation. Surface masks were usable on 148/150 frames; unusable masks leave candidates unchanged. SAM3 inference took 27.3 seconds on an H100 for these 150 frames.

With the original verifier and 2.5% margin, the visible subset improves from 28 correct/four wrong/one missing to **29/three/one**. The zero-margin version stays at 28/four/one. Both reject 32 of the 52 audited wrong original locations, but also reject one of the three newly confirmed balls—an airborne ball above the rink. Outputs on the 116 unresolved frames fall from 88 to 47, whose replacement positions are not automatically considered correct. The replay-trained verifiers plus surface filtering give 25 correct/two wrong/six missing. This is a post-audit hypothesis with two reported margins, not independent validation; the five-minute default replay is unchanged. `results/surface-filter-evaluation.json` contains the predicted polygons and all scores. A surface mask is useful context, not proof of ball visibility or metric calibration.

## SAM state-transfer failure

The pinned SAM3 predictor uses a nonblocking transfer to CPU for predicted masks when `offload_state_to_cpu=True`, then consolidates temporary outputs using those CPU tensors. On identical input frame hashes, seed coordinates, weights and crops:

| Runtime | Valid starting masks / 29 | Largest seed error |
|---|---:|---:|
| CPU state, original | 15 | 232.6 px |
| CPU state, explicit evaluation mode | 6 | 234.8 px |
| GPU state | 29 | 0.84 px |
| GPU state, repeat | 29 | 0.84 px |
| CPU state, synchronized transfers | 29 | 0.84 px |

The two GPU runs and synchronized CPU control produce identical coordinates for all 377 outputs. The constructor was already in evaluation mode; calling `.eval()` is not the causal fix. Production now keeps state on GPU. Synchronization wrappers are diagnostic controls only; no downloaded vendor source is committed or patched for production.

These are seed-consistency and repeatability checks, **not ball-detection accuracy**. See `results/sam-state-ablation.json`. The new short-track runner uses GPU state and a separate output directory. The original runner remains frozen because its code hash is part of the historical artifact provenance; use the new runner for this fix.

## Sequence sightings and temporal corroboration

GPT-6 Astra receives chronological whole-frame context and enlarged native crops. Five-second windows request ball sightings at 5 Hz. The prompt distinguishes visible pixels from an occluded inference and asks for the actual ball, not the crop center. Sixty main windows completed, producing 304 visible proposals. A separate discovery pass searches alternative detector regions where the original crop was misplaced: eleven completed windows, 119 proposals, with one remaining window withheld by the spend cap.

All 304 main proposals were visually reviewed after prediction: 165 showed an identifiable ball at the proposed point, 139 remained unresolved. Unresolved does not mean proven false. The audit neither moves coordinates nor filters the automatic full run. A proposal from an expensive language model is still only a proposal.

Each proposal starts an independent 13-frame SAM track, forward and backward. The final replay combines 6×6 native-pixel box prompts in camera-compensated 512×384 and 768×576 crops. The initial experiment used 10×10 prompts. A starting mask must stay within 12 pixels of its prompt. A sighting is admitted only if a local heatmap/verifier agrees, or tracks restarted at distinct nearby sightings return to each other's prompt. Intermediate masks need agreement from sightings on both sides, or a nearby sighting plus detector corroboration. Duplicate prompts in the same frame count as one temporal vote.

Old visual detections remain anchored. Short camera-compensated interpolation stays labeled `inferred`; older possession guesses remain `context`. Camera edits interrupt linking. Each frame has at most one output point. Agreement is not a calibrated probability and can still agree on the wrong object.

The expanded RGB model participates as a local corroborator selected on development data; it does not replace the existing YOLO detector. The no-label fusion implementation is `hockey_tracking/ball_refinement.py`, with the initial parameters in `fusion-config.json` and the development-audit amendment in `fusion-config-v2.json`. The amendment rejects appearance scores below 0.02, chooses the strongest appearance center within an agreeing cluster, and retains held-ball outputs as context. Short interpolation cannot overwrite a newly supported held-ball position.

## Prompt and transfer checks

The 29-seed prompt comparison predicts other reviewed frames, never its own prompt: 70 directed comparisons within short windows. Results are conditional on an already supplied ball sighting, and the checks are correlated development evidence.

| Tracker and prompt | Correct within 12 px | Wrong | Missing |
|---|---:|---:|---:|
| SAM3 box 10 | 61 | 2 | 7 |
| SAM3 box 6 | 61 | 0 | 9 |
| SAM3 box 14 | 63 | 3 | 4 |
| SAM3 point | 49 | 0 | 21 |
| SAM3 point + negative points | 59 | 2 | 9 |
| SAM3 circle mask, radius 3 | 60 | 2 | 8 |
| SAM3 circle mask, radius 5 | 61 | 3 | 6 |
| SAM2 large, box 10 | 58 | 7 | 5 |
| SAM3 box 6, wide crop | 65 | 2 | 3 |
| SAM3 box 6, close crop | 55 | 2 | 13 |
| SAM3 box 6, motion crop | 59 | 2 | 9 |

The two wrong wide-crop outputs had appearance scores 0.0023 and 0.0215; one passes the final 0.02 gate. Wider crops help retain a fast-moving ball, but cannot be treated as reliable without corroboration. The motion-crop experiment used nearby guide sightings and therefore had additional evidence. The early TAP pilot was too small for a general comparison.

The transfer protocol uses only YOLO detections three frames before and after each of the same 120 fixed review times. Reference coordinates and the target frame's detector output never prompt SAM. Consensus-only, appearance-selected, and YOLO/consensus hybrid policies are specified before opening their scores. This reuses already scored clips and is explicitly exploratory.

On the 27 visible transfer samples, automatic neighbor-seeded SAM consensus gave eight correct, three wrong, 16 missing. The YOLO/consensus hybrid stayed at YOLO's 22 correct, four wrong, one missing. A three-detector majority fallback gave 21 correct, five wrong, one missing. Neither is a demonstrated improvement.

A later motion-only reranking check used YOLO candidates at five neighboring times (-6, -3, 0, +3, +6 frames), directly registered to the target camera view. Four predeclared settings reward short, approximately constant-velocity paths, while preserving the target detector candidate and its frozen threshold. All four tied YOLO at 22 correct, four wrong and one missing on the same 27 visible samples. This approach does not create a missing target candidate. The numeric candidates, affine transforms and ranking decisions are included in `results/motion-selection-inputs.json` and can be recomputed without video. This was a follow-up on opened samples, not a fresh test.

## Language-model verification and failed coverage gains

An eight-request disagreement test used GPT-6 Astra with chronological full frames and candidate crops. Requests were selected by disagreement between YOLO and the old RGB detector, without reference coordinates. On the already opened 120-frame sample it corrected one held-ball case, giving 23 correct, three wrong, one missing instead of 22/four/one. Only two of the eight requests had evaluable visible-ball references. Four outputs on unresolved frames were removed, but their correctness cannot be scored. This is an exploratory one-frame gain, not independent validation. See `adjudication-protocol.json` and `results/adjudication-evaluation.json`.

More detections initially made tracking worse. Agreement among the new heatmap models proposed 142 additional seeds. The unfiltered fusion reached 792 detected/tracked frames, but a fixed 60-frame output audit found **12 wrong objects**, including scoreboard digits and pink stick tape. That replay was rejected. The public record preserves the failure in `output-audit-detector-ungated.json`.

Every additional seed is instead checked using a full target frame and chronological crops around the exact proposed point. GPT-6 Astra must distinguish a visible ball from a known non-ball or an uncertain image. Only a completed `visible_ball` verdict can enter SAM fusion; coordinates never move during this gate. Of the 142 heatmap seeds, 30 were retained, 97 rejected as non-ball, five uncertain, and ten withheld after an interrupted request. A separate YOLO proposal pass applies the same verification. These are automatic development filters, not new reference labels. Public verification reports preserve every verdict, source hash and retained coordinate. Incomplete or ambiguously interrupted requests are not silently retried.

The YOLO pass proposed 59 additional seeds: three were retained, 52 identified as non-ball and four uncertain. The final replay contains **417 detected, 203 tracked, 2,796 context, 219 inferred and 865 unknown frames**. Detected/tracked coverage is 13.78% instead of 9.02%; total estimated coverage is 80.78% instead of 79.73%. A fixed audit of 60 changed, non-seed tracked outputs confirmed 41 visible balls and left 19 unresolved. None of those audit verdicts enters fusion. This is development evidence, not independent accuracy.

The 94 reused approximate reference points were also reviewed with adjacent frames: 72 confirmed, 21 unresolved, and one disputed non-ball at frame 1912, where the reference appears to follow a lifted shoe. The original references remain intact. `reference-quality-audit.json` records the dispute, and the saved agreement counts are historical agreement only. We do not filter the reference retrospectively to report a higher accuracy.

## Distilling the sequence verifier

To test a cheaper alternative, 182 completed ball/non-ball decisions from the seed verifier became model-generated training labels: 33 positive and 149 negative. Chronological splits have 121 training, 33 development and 28 later test proposals, with a 12-frame boundary purge. The encoder remains frozen; two classifier heads are trained for 48 epochs: one on the target crop, another on camera-aligned preceding/target/following features. Epoch, threshold and variant selection use development F1, with conservative ties. Labels from the two transfer recordings are never used for this training.

The development-selected temporal head rejected all 24 negative later proposals but also missed all four positive ones. Applied to the frozen YOLO points in the 150-sample transfer check, it gave 14 correct, zero wrong and 19 missing instead of YOLO's 28/four/one. The image head gave 15/zero/18. This reduces false outputs by discarding too many true balls, and neither head is promoted. The same-game test measures teacher-label agreement, not independent truth; transfer uses previously opened references. `results/distillation-evaluation.json` includes all scores, thresholds, training history, weights hashes and per-frame outcomes.

A follow-up mixed the original 6,259 training crops with 121 teacher training examples (6,380 total). Each 64-example batch contains 24 original positives, 24 original negatives, eight teacher positives and eight teacher negatives. Two 20-epoch runs update either the head only or the encoder plus head, with frozen batch-normalization statistics. Checkpoint and threshold selection use the mean of original-validation and teacher-development F1. The encoder uses learning rate 1e-5; the head uses 1e-4. Both keep three of four later positive teacher cases missed, so neither fixes the late-segment failure.

| Verifier follow-up on all 150 samples | Correct / 33 visible | Wrong | Missing |
|---|---:|---:|---:|
| Original YOLO + verifier, recomputed control | 28 | 4 | 1 |
| Head replay training, rejection gate | 21 | 2 | 10 |
| Encoder replay training, rejection gate | 22 | 2 | 9 |
| Head replay training, candidate reranking | 24 | 5 | 4 |
| Encoder replay training, candidate reranking | 24 | 5 | 4 |
| Full1280 + full1920 + tiled640, original verifier | 27 | 5 | 1 |
| Same multiview proposals, head replay verifier | 26 | 4 | 3 |
| Same multiview proposals, encoder replay verifier | 24 | 7 | 2 |

The gates use their development-selected classification thresholds. Reranking retains the original detector operating point (detector confidence × verifier score ≥ 0.01), without a threshold search. Multiview fusion uses the existing implementation, with each tile family contributing at most one vote. None is promoted. These follow-ups happened after inspecting earlier failures and use the already opened test frames. They are useful failure diagnostics, not independent evidence of a new model win.

Conservative total OpenAI estimate: **$46.95**, including reserved cost for interrupted/incomplete requests, below the $50 campaign cap. Modal GPU compute is separate. No additional language-model calls are needed to reproduce the published fusion from its saved inputs.

## Reproduce

The Modal volume is `sport-scans-experiments`, with experiment output under `ball-hockey/reliability-20261018/ball-temporal-20261026-v1`. Source videos and frozen weights must already exist there; the protocols and worker code specify paths and source IDs. Install the Modal SDK and authenticate your own account. SAM3 requires approved Hugging Face access and cached licensed weights. OpenAI use requires your own authorized key; keys are never stored in this repository.

```bash
export MODAL_IMAGE_BUILDER_VERSION=2025.06
modal run experiments/ball-temporal-20261026/modal_run.py --help
modal run experiments/ball-temporal-20261026/modal_tracking.py --task prepare --batch dense-full-gpu --no-pilot
modal run experiments/ball-temporal-20261026/modal_tracking.py --task sam --batch dense-full-gpu
modal run experiments/ball-temporal-20261026/modal_tracking.py --task sam-prompts --batch dense-gpu
modal run experiments/ball-temporal-20261026/modal_tracking.py --task transfer
python experiments/ball-temporal-20261026/gate_detector_seeds.py
python experiments/ball-temporal-20261026/gate_detector_seeds.py --source yolo
python experiments/ball-temporal-20261026/derive.py --tracker sam3-box6 \
  --batches dense-full-gpu,dense-discovery-gpu,dense-wide-full-gpu,dense-wide-discovery-gpu,dense-detector-verified-gpu,dense-yolo-verified-gpu \
  --config fusion-config-v2.json --name final
python scripts/verify_ball_temporal_results.py
```

`derive.py` expects downloaded, complete JSON artifacts under the ignored build directory and verifies their seed hashes. Published references and evaluations must not be overwritten to claim a fresh test. The OpenAI runner records response IDs before completion and does not silently retry ambiguous failures. Cost estimates are conservative planning estimates, not invoices.

Research references informing the temporal experiments: [TrackNetV4](https://github.com/TrackNetV4/TrackNetV4), [TrackNetV5](https://github.com/thaonan/TrackNetV5-SDK), [TOTNet](https://github.com/AugustRushG/TOTNet), and [BlurBall](https://cogsys-tuebingen.github.io/blurball/). These are architectural references; our fine-tunes are not implementations or benchmark replications of those systems.
