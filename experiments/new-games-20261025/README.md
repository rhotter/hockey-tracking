# New NBHL recordings — October 25, 2026

Frozen-model comparison on two previously unused recordings from the official NBHL channel: Pittsburgh Legends finals, Evans City Taboo vs. Bloomfield Muckers, game 2; and South Jersey East Tier 3 finals, Trash Pandas vs. Jersey Dinos, game 1.

The recording IDs are absent from the prior 14-source registry. This is recording-disjoint testing, not guaranteed team- or venue-disjoint testing. Both videos were downloaded in full to the existing Modal volume. Inference covers fixed intervals at recording 10:00–11:30 and 30:00–31:30 in each game: **six minutes / 5,400 frames at 15 fps**, not two complete-game evaluations. Intervals were selected before viewing footage and retained regardless of outcome.

## Results

| Method | Correct / 145 | Wrong pair joins | Unverified joins |
| --- | ---: | ---: | ---: |
| SAM2 + OSNet + Qwen 2B | 116 | 6 | 88 |
| RF-DETR + Roboflow BoT-SORT | 96 | 7 | 16 |
| YOLO + Roboflow BoT-SORT | 98 | 5 | 11 |
| SAM2 without reconnecting | 108 | 0 | 23 |

Reconnecting preserved every box but added 65 unverified joins. Seven remaining positive checks lack a matched endpoint, and 22 have different IDs. The six wrong pair joins involve teammates. A separate review found repeated scene resets during continuous close-view play at Pittsburgh 10:21–10:22, preventing consideration of seven known same-player pairs across that boundary. No settings or reference labels were changed after inference.

The values are pair checks, not counts of independent identity-switch events. All 16 reports were independently recounted from exported sampled predictions; the reference seal and unchanged inference-code hashes passed verification.

## What was fixed

`data/protocol.json` fixes the sources, intervals, sample frames, checkpoint hashes, Qwen revision and four methods before raw-frame review. No training, threshold selection, or team-cluster fitting used these recordings.

- Existing NBHL YOLO → detector-supported SAM2 Large → generic OSNet body-region features → Qwen3-VL Embedding 2B joint-image blend and global forward assignment.
- Existing NBHL RF-DETR medium/960 → Roboflow BoT-SORT, using the defaults selected in the earlier development experiment.
- The same YOLO detections → the same Roboflow BoT-SORT.
- SAM2 local fragments without reconnecting, to isolate the contribution of the OSNet/Qwen stage.

All four methods receive the same scene-reset signals. The Roboflow adapter resets its tracker on those signals and emits associated current detector boxes.

The original SAM2, feature, candidate-selection and global-linking helpers are copied unchanged; their original hashes are in `data/inherited-source-hashes.json`. `scripts/new_games.py` adapts them to these new inputs. `data/code-freeze.json` records all experiment source hashes at the reference seal. Each remote inference stage records its first start time and the reference, protocol and adapter hashes. The RF-DETR and tracker environments are separate because their Supervision requirements differ.

## Reference and interpretation

The assistant drew boxes and reviewed identities directly on 24 raw frames, without detector proposals or tracker predictions. **No independent human verification has occurred.** There are 177 selected visible appearances, 145 known same-player pairs, 2,741 known different-player pairs and 1,018 uncertain pairs. These are correlated checks, not independent trials or continuous MOT ground truth.

Visible equipment, jersey evidence and short-term continuity support local identity groups. The same shirt color alone does not establish identity. Different people seen together establish a negative pair; known opposing teams also establish negatives. Ambiguous cross-time associations remain unknown. Groups are local to a clip and are not verified roster identities. The reference is selected observations, not exhaustive detection labels, so missed-reference counts do not measure detector precision or whole-frame recall.

`visual-notes.json`, `source-review.json` and `reference.json` were sealed before inference. Each sampled raw frame is bound to its extraction reservation by SHA-256. Boxes describe the visible body, excluding sticks and shadows; hidden positions are not invented. The webpage displays reference-only green-box photos separately from automatic output.

The evaluator uses one-to-one box matching at IoU ≥ 0.5. It counts recovered known same-person pairs, joins between known different people, joins with uncertain identity, and matched reference appearances. Results also separate pairs at most two seconds apart from more widely separated pairs. A larger temporal separation does not prove that a player was occluded in between. No IDF1, HOTA, full-game identity accuracy, player time, distance, shots or possession claim follows from this test.

## Reproduce saved counts

From the repository root, after installing the package:

```bash
python scripts/verify_new_games.py
python -m unittest discover -s tests -p test_new_games_reference.py -v
```

The verifier independently recomputes assignments and all pair outcomes from sampled predictions, checks the reference seal and source separation, and checks that sampled replay geometry agrees with the scored output. Results and sampled predictions are in `web/data/reliability/new-games-20261025/`; full interval box sequences are in `web/media/new-games-20261025/`. Replay video and review images are separate checksum-pinned website release assets, excluded from Git history.

## Qualitative failure review

The first Pittsburgh interval produces six scene-reset flags between recording 10:21.33 and 10:22.00. Three sampled raw frames around this boundary show continuous close-view play with blue #96. Seven known same-person pairs cross the boundary, which the frozen linker disallows. Fine patterned flooring and camera motion are possible triggers, but this test does not isolate their contributions. This post-inference review is recorded separately in `data/diagnostic-scene-review.json`; no reference labels, predictions or settings were changed.

## Run inference on Modal

This requires the existing private model/cache volume `sport-scans-experiments`. The trained detectors and full raw videos are not distributed with this repository. External model licenses still apply. Inputs are read from `/data/raw/<source-id>.mp4`; outputs are under `/data/ball-hockey/reliability-20261018/new-games-20261025-v1/`.

```bash
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run -e main experiments/new-games-20261025/scripts/modal_new_games.py --task detect
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run -e main experiments/new-games-20261025/scripts/modal_new_games.py --task track
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run -e main experiments/new-games-20261025/scripts/modal_new_games.py --task sam
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run -e main experiments/new-games-20261025/scripts/modal_new_games.py --task embed
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run -e main experiments/new-games-20261025/scripts/modal_new_games.py --task export
```

Track and SAM stages can run independently after detection; embeddings require SAM output. Each GPU stage uses an L40S, at most two containers per function. Preparation and exports use CPU containers. The preparation command requires transient YouTube metadata containing format URLs; these URLs are not published. Initial Modal CDN requests returned HTTP 403, so the full files were relayed through temporary local storage, verified against the remote SHA-256, then deleted locally. All frame extraction, video encoding and model inference ran on Modal.
