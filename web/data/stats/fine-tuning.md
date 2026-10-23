# Ball-hockey training results

Updated October 18, 2026. Only NBHL ball-hockey recordings are used as task training, evaluation and demonstration footage. Generic pretrained visual models provide the starting weights.

## What fine-tuning improved

The first selected player detector used 196 complete training frames from California, New Jersey and Georgia, containing 1,434 visible players and goalies. Pittsburgh remained the validation recording. We froze weights and confidence thresholds before each reported transfer evaluation.

On 36 complete Maryland and Halifax frames containing 256 players, generic YOLO11m achieved 81.9% precision and 63.7% recall. The NBHL fine-tune achieved 94.7% precision and 84.0% recall. Twelve ambiguous sampled frames were excluded.

On 22 complete Buffalo and North Shore frames containing 176 players, the selected NBHL YOLO11m found 149 players with six false detections. RF-DETR medium found 160 with 12 false detections. Medium was the primary RF-DETR candidate selected on Pittsburgh validation; the test does not select a new winner. These recordings are now regression data for subsequent development.

An additional indoor Buffalo game contributed 11 complete frames and 97 visible players. Both added YOLO variants finished training on 207 frames and 1,531 players; neither beat the original YOLO checkpoint on Pittsburgh validation. All six detector specifications were then frozen before viewing a Great Britain recording. On 14 complete frames containing 106 players, RF-DETR medium found 101 with six false detections, versus 90 with four for the original NBHL YOLO and 81 with 13 for generic YOLO. The two Buffalo variants found 91 and 93 with five false detections each. Two ambiguous frames were excluded before inference. This is a small assistant-labeled transfer test, not full-game reliability.

A later RF-DETR fine-tune also used the added Buffalo frames. It found 162 of 176 players with three false detections on the already-reviewed Buffalo/North Shore set, versus 160 with 12 before, but Great Britain coverage fell from 101 to 98 of 106 players. A separate goalie class did not improve Pittsburgh validation. These are regression results, not another untouched test.

## What remains unreliable

Detection is not identity. In a deliberately difficult sample of 82 long same-ID associations, 11 were supported matches, 39 were clear swaps and 32 could not be judged. A uniform contradiction check reduced those 39 swaps to 15, while retaining 10 of the 11 supported matches. That is useful but insufficient for player time on floor or full-game totals.

The ball detector uses 92 visible-ball training frames and 16 development frames. It improved localization in several transfer checks, but the ball could not be confidently located in most sampled frames. Unclear visibility is never counted as an empty negative label. The reported visible-frame precision does not measure whole-game false alarms or possession accuracy.

Vision-language models help review jersey crops and suspicious associations. They also invent readings on partial scoreboards and sometimes misread visible jersey numbers. Repeated answers can repeat the same error. Model-generated labels require review; agreement alone does not verify identity.

The rink replay uses approximate floor geometry and conservative foot estimates. It omits positions when the camera transform or feet are unavailable. The rink dimensions are not verified, so no miles or speed are reported. Camera-visible fragments are not complete shifts.

## Data and evaluation requirements

All labels in these experiments were reviewed by the assistant, not an independent human. Independent full-frame and contiguous-clip annotation is the next requirement for a dependable product. Split by entire game and preferably by venue; do not mix nearby frames across a claimed independent train/test split. Preserve unknowns and exclusions, audit goalie/field-player swaps, and evaluate same-team identity errors separately.

Official goals, assists and player names remain explicitly separate from visual predictions. The website includes three games and 24 reviewed goal clips. Shot totals, possession, distance and time on floor remain unmeasured.

[Measured results and downloadable audits](../../tracking.html)
