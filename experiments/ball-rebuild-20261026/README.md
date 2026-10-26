# Ball tracking rebuild — October 26, 2026

This experiment rebuilds the ball overlay in the existing five-minute NBHL
replay. All task-specific training and test footage is ball hockey. Player
tracks, the source video, and the original ball baseline are preserved.

## Pipeline

1. Run the existing NBHL YOLO11m ball detector on full frames at 1280 and 1920,
   and eight overlapping native 640-pixel crops. Merge nearby centers, allowing
   at most one vote per input view. Scores are evidence, not probabilities.
2. Verify candidates with a newly trained two-scale ResNet18 classifier. The
   selected spatial model preserves stride-four features and pools the candidate
   center as well as its surroundings. It sees native 64×64 and 192×192 crops.
3. Reject candidates on confident body/shoe regions from a generic COCO pose
   model run on full frames and crops. Reject extended red rink markings and
   large elongated warm-colored components. The latter targets stick tape:
   area ≥100 pixels, major-axis standard deviation ≥7 pixels, axis ratio ≥2,
   and distance to the component ≤3 pixels. These are imperfect floor/shape
   priors, not universal rules about balls.
4. Optimize a single offline path through the remaining candidates and an
   explicit unknown state. Camera-compensated displacement, visual evidence,
   track births, body overlap and rink geometry contribute to the score.
5. Bridge at most 0.2 seconds between compatible visual anchors, with assumed
   image uncertainty radius capped at 24 pixels. Do not extrapolate one-sided
   tracks. Image detections survive unavailable floor calibration.
6. Ask GPT-6 Astra to review short candidate episodes using full-frame context
   and enlarged native crops. Each episode is at most 30 frames and supplies
   up to five sampled positions; a singleton also supplies adjacent context
   frames. No annotation labels or previous model answers enter the prompt.
7. Retain an existing point only when its nearest sampled review says ball.
   Missing, failed, other-object and uncertain reviews withhold output. Never
   generate a new position from the language model. Held, airborne or unclear
   support has no overhead coordinate. Inferred gaps additionally require both
   original visual anchors to survive as accepted, mapped floor detections.
8. Recover additional visual candidates with a lower crop gate (0.01) and
   path evidence floor (0.08), only where the conservative CV pass was unknown.
   Review these episodes separately. After a diagnostic caught red background
   lights misclassified as airborne balls, require both reviewed floor support
   and a valid floor projection for this weaker pass. It cannot overwrite a
   primary point or introduce inferred positions. This sacrifices airborne recall.

The original ball detector checkpoint is unchanged: SHA-256
`8a518ae35c0c2a910dc9aa550ff9d60d827010f85f2fa4b97b6ab48f19244d7b`.
The new crop classifier is fine-tuned, not merely a prompt change. It uses 115
positive frames and 6,144 negative crops, including 23 current-game development
positives. See [training details and four-model ablation](training/README.md).
The spatial checkpoint is
`88d18c801ca34dfbe1decdbc4b506acfccad3f43a02e507b49f232722f00700a`.
Its selected rejection threshold is 0.9; the temporal evidence floor is 0.15.

## Why the first version was rejected

Native-resolution detection and temporal association produced 389 detections
and 77 inferred positions. A small nine-position check improved from 8/9 to
9/9, but a broader 36-output audit found nine wrong locations: shoes, stick
shaft tape, a fence fitting, a helmet and an inference that drifted away.
That version was not published as the fix.

Four crop-verifier variants and the shape check removed all nine diagnosed
errors while keeping 17 of 18 positive diagnostic examples. A subsequent audit
of 36 additional positions found 24 visible balls, two more shoe errors and ten
unclear cases. It also exposed a held ball with an unsupported floor projection.
These failures motivated the OpenAI review stage. Both audits are now development
evidence; they are not independent final tests. The first audit is in
`output-audit-v1.json`, the subsequent one in `output-audit-v4.json`.

## Evaluation scope

Source: `OLpK8enSjJs`, recording 900–1200 seconds, 15 fps, native 1920×1080.
This is an already-used recording, not a fresh independent game.

- Sixty uniform raw frames were fixed before new detector inference. Five had
  visible balls, one was an edit with no ball, and 54 were visually unresolved.
- Thirty additional raw frames were selected in two action windows before model
  scoring. Twelve had visible balls and 18 were unresolved. This sampling favors
  action and is not uniform game coverage.
- The original split had eight early development positives and nine later
  positives. The early points now train the verifier; later review and error
  analysis make the nine-point result a reused regression check.
- The old 16-position New Jersey set selects checkpoints and single-frame
  thresholds. Fifty-nine existing visible-ball labels from Maryland, Halifax,
  Buffalo and Nova Scotia remain outside verifier weight training.
- All labels are assistant-reviewed, approximate, and **not independently
  verified human ground truth**. Unknown visibility is excluded, never silently
  relabeled as a negative. A localization hit is within 12 native pixels.
  A wrong output on a visible-ball frame counts as both a false positive and miss.

The baseline's separate 36-output audit contains 33 visible ball positions and
three unresolved ones. Its outputs cluster in a few clear sequences. The v4
additional-position audit samples a different, harder distribution; the two
counts must not be presented as a controlled precision comparison.

## Cross-game detector comparison

This evaluates one proposed location per selected visible-ball image, before
pose/shape masks, temporal association, or OpenAI episode review. It does not
measure false positives during occlusion, stoppages or ball-absent footage.
Thresholds are selected separately on the 16 old development images.

| Input / model | Correct / 59 | Wrong location | Missed ball |
| --- | ---: | ---: | ---: |
| Full frame 1280 | 50 | 7 | 9 |
| Full frame 1920 | 54 | 3 | 5 |
| Native 640 crops | 51 | 6 | 8 |
| Crops enlarged to 960 | 53 | 5 | 6 |
| Multi-view ensemble | 54 | 4 | 5 |
| Ensemble + spatial crop verifier | 53 | 4 | 6 |
| RGB heatmap | 46 | 8 | 13 |
| Three-frame heatmap | 42 | 16 | 17 |

Higher resolution helped this regression check. The heatmap alternatives did
not win. The crop verifier helps reject some current-game objects but does not
improve this cross-game single-frame table. No generalization advantage for the
complete new pipeline has been established on a fresh game.

## Replay outcome

The final replay contains **284 visual detections, 20 inferred positions and
4,196 unknown frames**. The original baseline has 106 detected and 62 inferred
positions. Visual detections increase 2.68×, while total output frequency rises
from 3.73% to 6.76%. These are counts, not accuracy or continuous tracking recall.

`proposals-replay.json` preserves the conservative CV path: 286 detected and 23
inferred positions. Primary OpenAI review retains 258 + 20. The recovery pass
reviews 160 additional visual candidates in 81 episodes. The model accepts 51;
requiring reviewed floor support and valid projection retains 26. The final
replay merges those 26 without altering primary coordinates.

Primary review completed 71/71 requests; recovery completed 81/81. The combined
conservative API cost estimate is $9.39 ($4.99 + $4.40), excluding Modal compute.
All decisions, costs, input hashes and unsuccessful variants are retained.

The final conditional output audit contains 48 positions: **45 visible balls,
three unclear cases, and zero clear wrong locations**. It combines 24 primary
outputs (22 visible, two unclear) and 24 recovery outputs (23 visible, one
unclear). The primary samples exclude ±3 frames around earlier annotations;
only six recovery samples meet that exclusion. Most points are strongly
correlated in time or near development evidence. No manual point overrides
follow this audit. It is not an estimate of full-game precision or recall.
The reused nine-point check improves from 8/9 to 9/9. See
`output-audit-final.json` and the separately preserved audit rounds.

A diagnostic also checked six held/airborne/unclear-height detections against
full-frame and adjacent-crop context; the visible balls remain image-only where
floor support is not established. The four red background lights diagnosed in
the weaker recovery pass are withheld by the floor-support restriction.

The [machine-readable report](../../web/data/ball-rebuild-20261026/report.json)
contains state counts, scoped localization scores, audits, hashes and limitations.

Coverage is output frequency, **not accuracy**. Most of the recording remains
unresolved. The replay does not establish possession, shots, goals, assists or
metric ball speed. Error ellipses are conditional localization/motion assumptions,
not calibrated confidence regions; they do not cover every wrong-object or
camera-registration error. Both the crop model and language model can miss real
balls or accept a lookalike. A ball on a marking or severely blurred in motion
may be rejected by the shape prior. Review is sampled, not frame-exhaustive.

The page includes a previous/improved switch, synchronized video magnification,
short rink trail, timeline, frame stepping and a 4,500-row CSV with review support.

## Reproduce

GPU inference and bulk image work run on Modal's `sport-scans-experiments`
volume, environment `main`. Source frames are under
`/ball-hockey/reliability-20261018/overhead-20261025-v1/frames`; this experiment's
artifacts are under `/ball-hockey/reliability-20261018/ball-rebuild-20261026-v1`.
Images and weights are not committed to Git. Dependencies are pinned in the
Modal runners. Each of the two OpenAI review experiments has a $20 cap with input-token preflight, cost
reservations, and no automatic retry of dispatched requests. Estimates are not
invoices and exclude Modal compute.

```bash
modal run -e main experiments/ball-rebuild-20261026/modal_experiment.py
modal run -e main experiments/ball-rebuild-20261026/modal_alternatives.py
modal run -e main experiments/ball-rebuild-20261026/modal_alternatives.py --dense
modal run -e main experiments/ball-rebuild-20261026/modal_people.py
modal run -e main experiments/ball-rebuild-20261026/modal_people.py --tiled
modal run -e main experiments/ball-rebuild-20261026/modal_appearance.py
modal run -e main experiments/ball-rebuild-20261026/modal_verifier.py --adapted --hard-mining --spatial
modal run -e main experiments/ball-rebuild-20261026/modal_verifier.py --task infer --adapted --hard-mining --spatial
modal run -e main experiments/ball-rebuild-20261026/modal_appearance.py --shape
# Download numerical outputs to build/ball-rebuild-20261026 before local export.
python experiments/ball-rebuild-20261026/evaluate_verifiers.py build/ball-rebuild-20261026 appearance-shape.json
python experiments/ball-rebuild-20261026/publish.py build/ball-rebuild-20261026
# Fresh API inference requires an authorized key in an ignored env file:
modal run -e main experiments/ball-rebuild-20261026/modal_vlm_review.py --env-file PATH_TO_IGNORED_ENV_FILE
modal run -e main experiments/ball-rebuild-20261026/modal_vlm_review.py --env-file PATH_TO_IGNORED_ENV_FILE --recall
# Saved-artifact verification needs no API key or GPU:
python scripts/verify_ball_rebuild.py
python -m unittest discover -s tests -v
```

`modal_experiment.py --task batched` also benchmarks fp16 full-frame batches.
The selected path uses the original-precision detector outputs. The committed
candidate table, pre-review replay and review ledger reproduce the final path
on CPU without source footage or model weights. Sparse detector scores and
verifier gating are also recomputed from the saved numeric predictions.

## Related primary sources

- [SAHI sliced inference](https://docs.ultralytics.com/guides/sahi-tiled-inference)
  motivates preserving small-object pixels. This experiment implements its own
  tiling and fusion, not the SAHI package.
- [TrackNetV4](https://github.com/TrackNetV4/TrackNetV4) motivates testing temporal
  visual evidence. No badminton or ice-hockey performance is claimed here; the
  tested heatmap weights are the earlier NBHL pilot checkpoints.
