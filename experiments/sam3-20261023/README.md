# SAM3 and SAM 3D Body experiments

This campaign tests the newly accessible official SAM3 checkpoint, the newer SAM 3.1 Object Multiplex checkpoint, and SAM 3D Body DINOv3 checkpoint on NBHL ball hockey. No model is fine-tuned here. The reports distinguish complete-frame detection, sparse identity-pair tracking, and pose-emission diagnostics.

- `sam3-protocol-20261023.json`: prompts, splits, thresholds and selection rules.
- `sam3-detection-selection-20261023.json`: development-selected detector prompt/threshold.
- `sam3-tracking-selection-20261023.json`: development-selected native tracking prompt.
- `sam3-supported-amendment-20261023.json`: detector-supported tracker experiment added after the weak standalone detection result and before tracking scores were inspected.
- `sam31-amendment-20261023.json`: later SAM 3.1 comparison, explicitly specified after SAM3 regression results were known, without another prompt sweep.
- `sam31-capacity-amendment-20261023.json`: 128-slot follow-up on the one-minute development clip after the original run hit 64 objects.
- `sam31-geometry-amendment-20261023.json`: paired comparison of original boxes, final-mask boxes and largest connected mask components, motivated by oversized regression boxes and selected on development.
- `sam31-geometry-selection-20261023.json`: final-mask boxes won the coverage tie-break but did not improve the primary objective; no geometry variant was promoted to regression.
- `map-feet-visibility-v1.json`: the existing 48-box visibility reference, frozen before SAM3 access.
- Python files: the executed experiment and publication source. These are archived lab entry points, not a standalone inference distribution. They require the original Modal volume, cached clips/detections/features and `modal_campaign.py`; shared evaluation/linking source is archived under `../reference/scripts/`.
- Public outputs and individual regression-pair outcomes: `../../web/data/reliability/sam3/`.

The native SAM3 test changes detections, masks and identities together. The hybrid retains the existing detector, mask-overlap assignment, confidence thresholds and scene resets. It uses a separate SAM3 state for each tracked object because the core predictor cannot add IDs after propagation starts; image features are shared. Native masks may persist behind occlusion and are not visibility ground truth.

Engineering fixes preceded the scored hybrid run: select Modal's 2025.06 image builder; resolve Modal's canonical volume paths; pass SAM3's required propagation direction/preflight; convert pixel boxes to SAM3 relative coordinates; use independent object states for arrivals. Failed setup/integration attempts have no reported accuracy. The earlier source hashes in the initial freeze/amendment describe the initial runners; final executed source hashes are recorded in the release manifest. For SAM 3.1, the shared session wrapper passes an unsupported `offload_state_to_cpu=False` argument; the adapter drops only that false default and rejects true state offloading. No labels, scoring rules or selected thresholds were changed to resolve those API errors.

Frame staging copies identical JPEG bytes from the persistent cloud volume to local worker storage. It changes loading time, not model input. Native development timing includes slow volume loading; regression timing uses local staging. These wall times should not be compared as a model-speed benchmark.

All recordings and assistant-reviewed identity references were reused from previous campaigns. They are not an untouched test set or independent continuous identity ground truth. Unknown joins remain unknown. Foot emission is not ankle accuracy, ground contact or metric distance. The site retains these limitations and the unsuccessful alternatives.

Code: [SAM3](https://github.com/facebookresearch/sam3/tree/7f473cfd983d5eb43040ff608f66bff0d7bdcb53), [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body/tree/b5c765a0d89d789985e186d396315e7590887b94). Checkpoints retain Meta's license and access conditions and are not redistributed here.
