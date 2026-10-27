# Ball reconstruction from the surrounding play

The previous published tracker emitted 284 visual detections and 20 short interpolations in 4,500 frames. This experiment adds sequence-level possession/pass hypotheses and learned video point tracking. It uses the same five-minute NBHL recording, `OLpK8enSjJs`, recording time 15:00–20:00. It is a development experiment, not a new-game benchmark.

[Public replay](https://hockey-tracking-phi.vercel.app/overhead.html?t=60.6667) · [Machine-readable results](../../web/data/ball-sequence-20261026/report.json)

## Published result

The website and CSV now show one selected ball location per frame, without alternative locations or uncertainty circles. `positions.json` is the corresponding simple output. It uses the highest-weight contextual hypothesis already selected by reconstruction, or the retained visual/video-track position. Frames without an estimate remain empty. The original multi-hypothesis results and region audits below are retained as experiment records; region containment is not point accuracy.

| Output | Frames / 4,500 |
| --- | ---: |
| Retained visual detections | 284 |
| Added appearance-filtered video tracks | 91 |
| Contextual region estimates | 3,193 |
| Retained short interpolations | 20 |
| Unknown | 912 |

Estimated coverage is **79.7%**, versus **6.8%** previously. The standalone context regions have a median assumed radius of 62.9 pixels and a 90th percentile of 79.7 pixels in the 1920×1080 footage. Stoppages contribute substantially to coverage. This is not a measured accuracy rate.

The language model did not receive 121 of the reused reviewed frames. It provides regions on 110 of them, containing 107 reviewed locations, with median center error 13.9 pixels. Neighboring frames and occasional visual anchors remain available, so this is an inter-sample reconstruction check rather than a test of prolonged invisibility. Three failed full-segment requests completed in the repair ledger. Conservative API estimates/reservations total **$34.76**, including failed and incomplete experiments; Modal compute is additional.

## Reconstruction

1. **Visual anchors.** Preserve the previously published, model-reviewed detections. Their source replay hash is frozen in both protocols. No new human labels seed the video trackers or the sequence model.
2. **Learned video tracking.** Compare [CoTracker3](https://github.com/facebookresearch/co-tracker) and [TAPNext++](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapnextpp/votsp2026) on six preselected anchors. Each gets up to 12 seconds of video, forward and backward from the seed, in a camera-stabilized 768×576 native-resolution crop. TAPNext++ at 512 input resolution performs better in this pilot and is also run on 31 anchors throughout the segment. Vendor commits and checkpoint hashes are saved. Both models run on Modal H100 GPUs.
3. **Ball appearance verification.** Score the temporal proposals with the existing two-scale spatial ResNet18 ball verifier. Raw tracker visibility is insufficient: a tracker can confidently follow a shoe or background after losing the ball. Compare five appearance thresholds on reused labels; freeze 0.3 for proposals agreeing with context, and 0.9 for proposals within 12 frames of a seed when context cannot locate the ball. Require at least three consecutive supported points and limit camera-compensated per-frame displacement to 120 pixels. These thresholds were selected during development.
4. **Sequence reasoning.** GPT-6 Astra, high reasoning effort, sees chronological samples at 2 fps over a 10-second core with three seconds of context on each side. It returns states every half-second: carried, passing, shooting, loose, contested, stopped, outside the view, or unknown. Inputs include player fragment labels and occasional existing visual proposals. It uses earlier and later action, stick motion, receivers, challenges and officials. It may infer an invisible ball, retain alternative locations, or leave the state unresolved. The ball's height support is classified separately.
5. **Dense reconstruction.** Between model samples, a carried ball follows the carrier's actual observed movement plus a changing ball/stick offset. Other regions interpolate in camera-compensated coordinates. Alternatives are matched and retained instead of averaged together. Regions have a minimum radius of 60 native pixels; hypotheses requiring more than 180 pixels are withheld. Minimum known weight is 0.55, and minimum dominant weight is 0.4. Weights and regions are subjective model estimates, not calibrated probabilities.
6. **Fusion.** Retain the original visual observations, refine additional locations with the supported video tracks, and use contextual regions elsewhere. Never promote the point tracker's latent occluded coordinates to visual observations. A known edit at frames 466–515 resets inference. Held, airborne and height-uncertain estimates remain in image space.

There is no new weight fine-tuning in this pass. The appearance verifier was fine-tuned in the preceding ball rebuild; the video trackers and language model use pretrained weights. The system is an offline reconstruction that uses future frames, not a causal live tracker.

## What the numbers mean

The final `report.json` separates visual, video-tracked, inferred and unknown frames. Estimated coverage includes contextual regions and stoppages. It is not visible-ball recall, localization accuracy, or accuracy of possession, goals, shots or assists.

The six-seed pilot favored TAPNext++: among answered reused visible-ball samples, 26/35 were within 25 native pixels, versus 8/30 for CoTracker3. Extending raw TAPNext++ to 31 seeds exposed substantial drift: 41/91 answered positions were within 25 pixels. Appearance filtering at 0.3 improved that reused check to 46/47, with a median error near 3 pixels. These are approximate assistant-reviewed locations, with seed frames and their ±3-frame neighbors excluded. The appearance threshold comparison uses the same development labels, so it is not independent validation or a full-game precision score.

The sequence-region check also reuses old visible-ball labels, mostly at positions already known to the previous tracker. It must not be presented as evidence that newly inferred invisible positions are correct. `audit-protocol.json` separately defines a systematic conditional audit of newly emitted contextual and tracked frames. The assistant examines full-frame before/after context and native crops, recording visible containment, visible misses, plausible but unverified context, or unclear evidence. An invisible ball has no independent ground-truth location here.

The final audit contains 73 newly emitted positions selected by that frozen protocol. Among 53 contextual positions, five show a visible ball inside the region, 38 fit the surrounding play but remain unverified, and ten are unclear. Among 20 added video-track positions, 13 show the ball inside the 24-pixel region, five remain unverified, and two are definite misses. At frames 4021 and 4026 the ball stops near an official while the tracker follows the nearby stick. These misses remain published; appearance filtering reduces drift but does not eliminate it. This conditional, correlated review cannot establish accuracy for the invisible portions of the game.

## API failures and accounting

Six initial non-streaming requests encountered a 180-second transport limit or exhausted the original 8,192-token output cap. A separate streaming run at that cap also produced incomplete structured timelines. The successful protocol uses streaming with a 900-second socket timeout and a 16,384-token allowance. Complete failure ledgers and reservations remain in `results/`; they are not omitted from cost accounting.

The extended pilot uses three workers. The remaining 24 windows use six workers in a separate ledger. Any explicitly launched replacement for a failed window uses a separate repair ledger with three workers, preserving the original failure. There are no automatic generation retries. The total API experiment allocation is $50; the report conservatively sums actual estimated usage or unresolved-request reservations, without counting the merged ledger twice. Estimates ignore cache discounts and do not include Modal compute.

## Reproduce

The numeric model outputs are saved under `results/`. Rebuild the replay and metrics without GPUs or API calls:

```bash
python experiments/ball-sequence-20261026/publish.py
python scripts/verify_ball_sequence.py
python -m unittest tests.test_ball_sequence -v
```

To run the cloud experiment against the existing Modal footage volume:

```bash
modal run experiments/ball-sequence-20261026/modal_points.py
modal run experiments/ball-sequence-20261026/modal_points.py --models tapnextpp512 --all-seeds
modal run experiments/ball-sequence-20261026/modal_points.py --verify
modal run experiments/ball-sequence-20261026/modal_context.py --env-file /path/to/private.env
modal run experiments/ball-sequence-20261026/modal_context.py --env-file /path/to/private.env --remaining-windows
```

`modal_context.py` reads an existing authorized key into an ephemeral Modal Secret. Credentials, source video, image crops, vendor code and model checkpoints are excluded from Git. The public site serves the existing five-minute review video, generated numeric results and explanatory overlays.
