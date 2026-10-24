# SAM3 adaptation and tracking combinations

This campaign fine-tunes SAM3 on NBHL ball-hockey boxes, transfers the detector into native video tracking, and tests combinations with the existing detector, SAM2, OSNet and Qwen image embeddings. It also tests visual prompts, native association settings and longer execution. The public reports keep unsuccessful alternatives and distinguish detection, sparse identity checks and runtime behavior.

## What was trained

196 completely reviewed frames from three NBHL recordings contain 1,434 player/goalie boxes. Horizontal flips produce 392 training examples. A fourth recording supplies 23 validation frames with 178 boxes. Two additional recordings supply 36 reused regression frames with 256 boxes. The six annotation files are included under `annotations/`; video and image files are not included here.

Six actual weight-training runs compare detection heads, the grounding decoder, the grounding encoder plus decoder, and a decoder with the final four visual attention blocks. Learning rates and the visual-backbone extension are recorded in the protocols. The selected model is `transformer-lr1e5`, epoch 2, confidence 0.5. It updates 22,230,581 parameters in the grounding transformer and scoring layers. The visual/text backbones, geometry encoder, segmentation head and temporal tracker remain frozen in that selected run. This is box-supervised detector adaptation, not temporal-tracker or mask training.

Training uses AdamW, Hungarian assignment, L1 and generalized-IoU box losses, focal classification loss, a fixed seed and horizontal flips. Frozen visual features are cached for the non-vision variants. The visual variant runs the backbone on images and substitutes differentiable linear/activation operations for an inference-only fused MLP while gradients are enabled. Validation retains the official inference implementation. Gradient checks, changed tensor counts, checkpoint hashes and full hashes of frozen parameters are saved in each report.

All six runs finished before checkpoint/threshold selection was frozen. Only that winner and the original model were then evaluated on the regression frames. The original checkpoint was a selection candidate. The paired box-only evaluation uses float32 sigmoid/presence calculations for both models. The prior campaign's processor rounded confidences to bfloat16; one identical box moves from 0.6015625 to 0.5983377695 across the 0.6 threshold, explaining 93 original matches here versus 94 previously.

## What was combined

- **Detector anchoring:** one-to-one overlap assignment between native SAM3 mask boxes and current YOLO player boxes. Only current detector boxes are emitted.
- **Appearance linking:** appearance/scene/gap checks and the existing global forward linking algorithm applied to anchored SAM3 fragments. The selected split threshold may disable appearance splitting; its exact setting is retained.
- **Consensus:** bounded SAM3 agreement evidence on existing SAM2 reconnection candidates, with OSNet alone or the frozen OSNet/Qwen joint-image blend.
- **Native settings:** association overlap, mask reconditioning interval and track lifetime.
- **Visual prompts:** an automatically chosen initial detector box, alone or with text. A separate experimental variant retains the initial geometry/appearance tokens across frames. It is an adapter to the pinned API, not a claimed official visual-template workflow. The upstream image-only prompt path otherwise stops admitting new detections after its initial geometric prompt.
- **Adapted video:** the frozen detector delta with hockey-player thresholds/admission margins. A later, separately registered development candidate retains the original person prompt and native thresholds. Two later capacity controls also test the previously trained head-only and decoder-only checkpoints with native person settings, after the full-transformer development runs disappointed. This is an explicitly adaptive development search registered before new tracking regression scores. The final adapted winner has an original-weight control with matching prompt and thresholds.

Nine development clips select a winner for each family using correct − 10 × wrong, then observation coverage and fewer unverified joins. Frozen choices are tested on four 30-second clips and a separate five-minute clip. These references were already used in earlier campaigns. Unknown reference identities stay unknown. No per-player manual correction is applied to displayed tracks.

The rink-surface filter is explicitly a **post-regression follow-up** motivated by bench-player false positives. It compares 18 automatically predicted surface-mask filters and the unchanged fine-tune on the 23 validation frames. Only a validation-improving filter is checked on the reused regression frames. There is no manual rink polygon. Packed masks and predictions allow every filtering result to be recomputed. A surface mask is not rink calibration or visibility ground truth.

A further fixed detection audit completely labels eleven Pittsburgh frames at recording 42:00–42:30 before viewing these SAM3/filter predictions: 77 visible players and goalies, including seven goalies. Existing YOLO proposals assisted boxes and one missing player was added manually. This makes the YOLO comparison biased toward its own proposals. All models, thresholds and the surface filter were already frozen; no selection uses this audit. The recording was previously used for tracking, so this is a new complete detection sample, not an independent game.

## Longer runs and audit

The 20m 15s replay uses four 315-second executions with 15-second overlaps. Existing full SAM2 runs with generic and trained appearance are also exported for qualitative comparison, without new inference. Every overlapping input JPEG is checked byte-for-byte. The joiner requires strong, unique, simultaneous geometric agreement; it does not invent positions across gaps or establish roster identity.

A four-frame image cache and CPU storage for old mask history allow longer execution. The full history remains available. All output IDs, boxes and scores match the original implementation across 1,200 control frames. A separate cache for immutable memory-eligibility decisions matches 48,000 synthetic selections and 1,200 complete output frames; it invalidates when object removal replaces a score tensor. Timing comes from different cloud workers and is not a controlled speed benchmark. CPU memory can be large even when GPU memory is reduced.

The new 42:00–42:30 audit labels four initially visible players once per second before viewing candidate tracking IDs. It has 88 visible, 29 out-of-view and seven uncertain observations. Both trackers have the same preceding context. It is a new temporal sample from a reused recording, with assistant-reviewed, detector-assisted boxes, not an independent game or continuous annotation. The earlier one-second audit is reported separately as a supplementary reused check.

## Reproduce the published measurements

From the repository root, install the project's Python dependencies, then run:

```sh
python scripts/verify_sam3_improvements.py
python scripts/verify_results.py
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

`web/data/reliability/sam3-20261024/` contains report hashes, all training/selection reports, regression predictions, pair outcomes, audit observations and packed surface masks. The verifier recounts detection, checks selection rules and frozen-weight evidence, reassigns exact sampled predictions to reference boxes and recounts identity pairs, recomputes the new audit, and validates replay frames. `scripts/build_sam3_improvements.py` renders the page from those reports. Replay video and review sheets are delivered through the checksum-pinned asset release, outside Git.

The `modal_*.py` files here are the executed lab entry points, not a standalone inference package. They require the lab's `modal_campaign.py`, approved Hugging Face access, the original Modal volume and cached recordings/detections/features. Shared earlier evaluation/linking source is archived under `../reference/scripts/`; the initial SAM3 adapter is under `../sam3-20261023/`. The initial live-session coordinator was replaced after a session disconnect by `modal_nbhl_sam3_resume.py`, a bounded detached cloud coordinator. It reuses completed runs and frozen selections, then performs the registered regressions and exports evidence. Long runs reserve 512 GiB host RAM after a lower-memory run was terminated; this changes resources, not inference rules. The cloud run completed before publication.

Code and checkpoint provenance: [pinned SAM3 source](https://github.com/facebookresearch/sam3/tree/7f473cfd983d5eb43040ff608f66bff0d7bdcb53), checkpoint revision `3c879f39826c281e95690f02c7821c4de09afae7`. Checkpoints and trained deltas remain on Modal and are not redistributed. Meta's model license/access conditions continue to apply.

These experiments do not establish reliable full-game identities, time on floor, miles run, shots, goals, assists or possession. Track labels remain automatic fragments.
