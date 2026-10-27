# Reproduction boundaries and external artifacts

## Included and runnable offline

- Pair outcomes and all 117 published count comparisons.
- Frozen prompts, selection records, model revisions and source hashes.
- A portable global reconnection component and synthetic example.
- Equivalence tests against the original global solver.
- SAM3 ball point/box prompt outputs, frozen pilot selection and exploratory amendments, mask/appearance scores, and deterministic single-position hybrid replays. `scripts/verify_ball_sam.py` reproduces the numeric comparison and audit selection without model access.
- Sequence-level ball-state outputs, CoTracker3/TAPNext++ numeric point tracks, appearance scores, API failure/cost ledgers and deterministic reconstruction. `scripts/verify_ball_sequence.py` reproduces the previous contextual overlay used by the SAM hybrid without model access.
- SAM3 and SAM 3.1 sparse pair outcomes and compact replay boxes, plus SAM 3D Body numeric predictions and visibility references. `scripts/verify_sam3_results.py` checks these without model access.

## Reference source

`experiments/reference/` preserves selected original experiment files byte-for-byte. Historical absolute `/data/...` paths describe the cloud experiment layout, not data included in this checkout. Research orchestration imports and local mounts may refer to the original laboratory repository. **Do not treat these archived Modal entrypoints as self-contained executable jobs.** They are included to audit how the reported features, candidate lists, policies and scores were produced.

`scripts/verify_results.py` reads the archived source without executing Modal or loading models. It writes an ignored `outputs/verification.json` report. The release adds no public access to the original Modal volume or private website.

Fresh end-to-end reproduction requires lawful access to the listed recordings, the original trained YOLO detector checkpoint, compatible SAM2 and OSNet weights, source frame alignment, reviewed reference observations, and regenerated candidate/crop/feature caches. The detector checkpoint and complete inference caches are not distributed. Generic model access alone is insufficient to reproduce the detector outputs in these experiments.

## Third-party projects

These references identify dependencies, not an assertion that all use the same license. Check their own license files and model cards before use or redistribution.

- [CoTracker3](https://github.com/facebookresearch/co-tracker) and [TAPNext++](https://github.com/google-deepmind/tapnet): learned video point trackers. Pinned revisions and checkpoint hashes are in `experiments/ball-sequence-20261026/`. No vendor code or weights are redistributed.
- [Ultralytics](https://github.com/ultralytics/ultralytics): YOLO detector tooling; the private trained checkpoint is not included.
- [SAM2](https://github.com/facebookresearch/sam2): mask propagation.
- [SAM3 / SAM 3.1](https://github.com/facebookresearch/sam3): pretrained concept segmentation and multi-object tracking; pinned code `7f473cfd983d5eb43040ff608f66bff0d7bdcb53`. Checkpoints require their own access and are not redistributed.
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body): DINOv3 body-pose audit; pinned code `b5c765a0d89d789985e186d396315e7590887b94`.
- [Selective Mask Propagation](https://github.com/holma91/selective-mask-propagation): source environment used by the research scripts.
- [Deep Person Re-identification / OSNet](https://github.com/KaiyangZhou/deep-person-reid): appearance features.
- [Qwen3-VL Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding): multimodal image embeddings. Pinned code revision: `393e2978d27852b0d0230d6994f37f9c15bed73c`.
- [Qwen3-VL Embedding 2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B): revision `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`.
- [Qwen3-VL Embedding 8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B): revision `2c4565515e0f265c6511776e7193b22c0968ddc7`.

The original recording links are in `results/sources.json`. The repository contains original numeric annotations and prediction outcomes, not the recordings or image crops.
