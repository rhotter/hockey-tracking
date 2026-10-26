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

The [sequence reconstruction experiment](../experiments/ball-sequence-20261026/) adds an offline ball-specific layer. GPT-6 Astra sees chronological frames before and after each target, inferring possession, transfers, stoppages and alternative ball regions. A carried ball follows the carrier's observed movement plus an estimated stick offset. TAPNext++ tracks visible point evidence over video windows; the previously trained appearance verifier rejects drift. Original visual observations take precedence, while contextual estimates fill supported hidden intervals with wider uncertainty regions.

These inferred ball states are separate from the player reconnection API. They are not observed player positions or validated game statistics. The public replay distinguishes visual, tracked, contextual and unresolved states. Error radii and model weights are uncalibrated; held or airborne balls have no floor projection. The five-minute demonstration remains a development recording, with conditional assistant review rather than independent human ground truth.
