# Ball context and uncertainty

Experimental extension of the five-minute NBHL overhead replay. The detector,
camera solution and 106 original ball anchors are unchanged. This adds an offline
motion model and possible player-occlusion context; it does not train a new model.

Run from the repository root with NumPy installed:

```sh
python experiments/ball-context-20261026/run.py
python scripts/verify_ball_context.py
python -m unittest discover -s tests -v
```

Optional cloud execution and native-frame review, using the existing experiment
volume:

```sh
modal run -e main experiments/ball-context-20261026/modal_run.py
```

## Method

1. Transform detection anchors into the existing camera-compensated rink plane.
   Preserve detector scores separately from motion uncertainty.
2. Bridge nearby observations using both endpoints. Ordinary gaps must be at
   most 0.4 seconds. If at least half the interpolated image positions overlap
   tracked player boxes, permit up to 0.8 seconds. This is possible occlusion,
   not proof that a player holds the ball.
3. Use the last two compatible anchors for brief constant-velocity prediction:
   at most 0.2 seconds in clear view, or 0.4 seconds with player overlap.
   A nearby contradictory future anchor suppresses prediction.
4. Propagate a 6-pixel localization standard deviation through the homography
   Jacobian. Add motion-process uncertainty; disagreement between incoming and
   outgoing motion contributes additional covariance to a bridge. One-sided
   prediction includes the correlation between the finite-difference velocity
   and its anchor position. Its uncertainty increases with time.
5. Withhold estimates across camera failures, outside the image/rink, after the
   gap deadline, or when the largest 2-sigma image radius exceeds 100 pixels.
   Never initialize a ball from player positions alone.

The ellipses are **assumed model uncertainty**, conditional on correct anchors,
camera calibration and a floor-level ball. They are not calibrated 95% confidence
regions and do not account for a false detector anchor, airborne motion or every
camera-calibration error. Covariance is in relative rink coordinates; the image
covariance and displayed radius are in native 1920×1080 pixels. No metre-based
accuracy is claimed. The displayed radius is the ellipse's longest semi-axis.

These are hand-set, inspectable model parameters. They were not selected by
optimizing the reported masking check. There is no learned possession model,
stick detector, LLM reasoning, multi-hypothesis identity model or 3D reconstruction
in this extension. Such context needs ball-hockey labels and independent tests.

## Results and evidence

Same development recording as the overhead demo, OLpK8enSjJs, recording 15:00–20:00,
15 fps. This is not a new-game test.

| Variant | Detected | Inferred | Unknown |
| --- | ---: | ---: | ---: |
| Original detector anchors | 106 | 0 | 4,394 |
| Forward motion only | 106 | 59 | 4,335 |
| Two-sided motion plus short prediction | 106 | 62 | 4,332 |
| Above plus player-overlap context | 106 | 62 | 4,332 |

The context variant has 29 ordinary bridge frames, 3 possible-occlusion bridge
frames, 28 motion-prediction frames and 2 possible-occlusion prediction frames.
**Player context did not increase coverage over the two-sided motion variant
on this segment.** The extra 62 estimates represent 4.13 seconds; total ball
output covers only 3.73% of frames. More output is not evidence of more accuracy.

We deliberately removed blocks of 1, 3 and 5 detection anchors in sufficiently
dense runs. There were 39 target-frame instances across these trials. Forward
prediction returned 33, with median/p90 disagreement of 3.29/17.25 pixels against
the hidden detector centers. Two-sided inference returned all 39, with
4.00/14.59 pixels. The variants cover different samples; compare the complete
trial records, not just those summary percentiles. All recovered detector
centers fell within the assumed ellipses, which may simply be broad. This is
**detector-consistency testing, not ground-truth ball accuracy or uncertainty
calibration**. The same base anchor filter previously used nearby observations;
that selection is not an independent held-out benchmark.

Native-resolution crops were exported on Modal for 15 deterministically selected
inferred frames. The assistant reviewed them without overlaid predictions and
recorded three clear approximate centers in `visual-review.json`. The remaining
frames were left unscored because blur, contact or occlusion prevented a
confident center assignment. Crops were selected around the predictions, so this
is explicitly biased, sparse development review. It does not establish recall,
false-positive rate, correct possession, shot accuracy or full-game reliability.
The unscored frames remain hypotheses, not verified positions.

The website reuses the synchronized public video. Green means a detector anchor;
orange and a dashed ellipse mean inferred; grey timeline regions are unknown.
Use the ball selector to compare with detections alone, and download the CSV
for states, coordinates, uncertainty radius and reasons. Original player data
and original ball results remain intact.

## Related research

[TrackNetV3](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf) separates
visual shuttlecock prediction from trajectory repair. Its badminton-trained
weights and accuracy do not establish performance on ball hockey.
[Ball Trajectory Inference from Multi-Agent Sports Contexts](https://arxiv.org/abs/2306.08206)
studies learned ball inference from player trajectories in soccer. It motivates
testing richer player context, but this implementation does not use that model
or claim transfer of its results.
