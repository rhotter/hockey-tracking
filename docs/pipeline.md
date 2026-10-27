# Pipeline and input contract

The measured pipeline samples NBHL footage at 15 fps, uses a ball-hockey-fine-tuned YOLO11m detector, and matches SAM2.1 Large mask predictions back to current detections. It creates temporary track fragments. The displayed observations remain detector-supported; the linker does not invent hidden positions.

OSNet represents player appearance with three horizontal feature-map regions. A quality heuristic favors confident, larger, less-overlapped crops, retaining up to twelve temporally spaced prototypes. Up to three crops from temporal groups of those prototypes are encoded together by Qwen3-VL Embedding 2B. The identity instruction and exact checkpoint revisions are preserved in the reference source and protocols.

Candidate construction is label-free. It checks temporal order, same scene, support and image-space motion after camera compensation. The image-model shortlist combines the top two successors/predecessors by appearance with every baseline-accepted edge. It requires OSNet cosine >= 0.65. Final assignment requires at least ten supported observations on each fragment and at most 300 frames between them.

For shortlisted edges, the best observed challenger uses:

```text
similarity = 0.5 * OSNet cosine + 0.5 * joint-image Qwen cosine
benefit = similarity - 0.75 - 0.001 * gap_frames / 15
```

The assignment maximizes total positive benefit, permitting unmatched fragments. Each fragment has at most one predecessor and successor. Strict temporal ordering prevents cycles and joining co-visible fragments. Similarities are not calibrated probabilities. Changing one edge can change other assignments.

## Portable API

`reconnect(result, evidence, candidates, fps=15)` accepts:

- `result`: a dictionary with `frames`, each containing `frame` and `objects`. Each object has a string `track_id`; any additional observation fields are preserved.
- `evidence`: keyed by those string IDs, with integer `start` and `end` frame indices for each fragment.
- `candidates`: dictionaries with `a`, `b`, `gap` (frames), `distance_heights` (camera-compensated distance divided by mean box height), `support_a`, `support_b`, `cosine_mean` (OSNet) and `cosine_joint` (Qwen).

The caller must construct valid same-scene candidates and truthful intervals/support counts. An absent/null `cosine_joint` means the edge was not shortlisted and excludes it from this blend policy. The core does not inspect images, apply camera compensation, or calculate embeddings. It rejects frame rates other than 15 rather than silently changing the measured behavior.

The output is a deep copy of the observations with connected IDs relabeled. `local_track_id` preserves the previous fragment ID and `links` records accepted edges. Frames, boxes and missing intervals are unchanged.

## From tracks to stats

The [Roboflow replacement experiment](../experiments/roboflow-20261025/) supplies an alternative detector/tracker adapter using RF-DETR and five upstream tracker implementations. Sports TeamClassifier and ViewTransformer can supply team clusters and coordinate projection, but they do not establish roster identity, camera calibration or game events. The tested Roboflow trackers and retained-OSNet hybrid underperformed the current pipeline on the reused identity checks, so the recommended path above is unchanged. The hybrid does not include Qwen embeddings. Team clustering also failed on many Maryland observations; its confidence gate is not a guarantee.

Full-game player identity requires roster/jersey evidence and memory across camera cuts and shifts. Time on floor requires entry/exit evidence, not merely camera visibility. Distance requires verified rink dimensions, camera calibration and reliable floor positions. Shots and possession additionally need ball/event observations. None of these is supplied by the reconnection API.

## Ball state from video sequences

The [ball refinement experiment](../experiments/ball-temporal-20261026/) adds a separate offline layer at 15 fps. The retained ball-hockey YOLO detector generates local candidates; an appearance model rejects some false objects. Nine new RGB/temporal/motion detector fine-tunes did not beat YOLO on the sparse new-recording checks.

GPT-6 Astra receives chronological full frames and enlarged native crops, distinguishing visible ball sightings from possession or hidden-ball guesses. Each sighting starts a short forward/backward SAM3 track. GPU-resident state avoids a demonstrated mask-transfer race. The selected replay combines tight box prompts at two crop sizes. A starting mask must match its prompt and have local detector support or reciprocal agreement with another sighting. Intermediate points need agreeing tracks from distinct seed frames or a recent sighting plus detector evidence. Additional heatmap/YOLO proposals must pass a separate sequence-level ball/non-ball verification first.

Original visual anchors remain unchanged. Supported held-ball positions stay contextual; short gaps may receive camera-compensated interpolation. Camera edits block linking. Existing possession estimates fill other intervals when available. The output contains one point or an unknown state per frame, and the interface distinguishes detected, tracked, inferred and contextual points. Agreement and appearance scores are not calibrated probabilities.

The diagram maps floor positions using the camera homography. A held, airborne or height-unclear estimate uses the identified carrier's feet when available, otherwise an approximate camera-ray/floor intersection. The diagram labels these approximate positions with ≈; they can be displaced from an elevated ball's actual horizontal location. Invalid camera alignment and off-rink projections have no diagram point. CSV exports retain the original coordinates and separate diagram coordinates/method.

These ball states do not establish player identity, time on floor, metric distance, possession percentages, shots or goals. The five-minute replay is a development recording; sparse assistant review is distinct from independent human ground truth or full-game accuracy.
