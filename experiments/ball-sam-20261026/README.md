# SAM3 for the ball

This experiment tests SAM3's temporal tracker on NBHL ball hockey, using the same five-minute Bellmawr Hawks / South Jersey Nights Watch recording and starting ball observations as the existing TAPNext++ experiment. It does not fine-tune SAM weights or make new language-model requests.

## Six-seed result

All methods are evaluated at the same 105 reviewed visible-ball positions, after excluding the seed neighborhoods. Counts below apply after the frozen filters; withheld positions are included explicitly.

| Method | Within 25 pixels | Outside 25 pixels | No output | Median error when emitted |
|---|---:|---:|---:|---:|
| TAPNext++ + appearance | 23 | 0 | 82 | 4.82 px |
| SAM3 point, 768 crop + filters | 27 | 0 | 78 | 1.17 px |
| SAM3 box, 768 crop + filters | 6 | 0 | 99 | 2.29 px |
| SAM3 box, 384 crop + filters | 2 | 1 | 102 | 1.10 px |

Unfiltered SAM is unreliable: point-768 makes 13 errors beyond 25 pixels among 41 emitted reference positions, and box-768 makes 35 among 42. Mask size and appearance filtering are essential. A low median alone hides these failures.

The frozen development split selects **box-768**: two correct development positions versus one for point-768. That choice fails to generalize even to the reserved seed windows: four correct positions versus TAP's 20. Point-768 gives 26 there, but choosing it after seeing this result is post-hoc. The [exploratory amendment](exploratory-amendment.json) records this before either 31-seed run. Both variants are expanded; point-768 must be described as exploratory. The evidence supporting configuration selection is particularly sparse.

## Larger 31-seed result

The larger runs use the same 31 seed frames as TAPNext++, with 103 eligible reused reference positions. The combined proposal count is after context gating, before retaining original anchors and requiring continuous runs in the final replay.

| Proposal method | Within 25 pixels | Outside 25 pixels | No output | Median error when emitted |
|---|---:|---:|---:|---:|
| TAPNext++ + appearance | 46 | 1 | 56 | 3.01 px |
| SAM3 box + filters | 45 | 1 | 57 | 1.30 px |
| SAM3 point + filters | 52 | 0 | 51 | 1.26 px |
| Box-SAM + TAP + context gates | 51 | 1 | 51 | 1.48 px |
| Point-SAM + TAP + context gates | 67 | 1 | 35 | 1.51 px |

The point hybrid changes 49 final replay positions and raises video-tracked frames from 91 to 122, replacing 31 contextual estimates. Overall output coverage remains **79.7%**. The improvement is more positions supported by a video tracker, not extra inferred coverage or proof of invisible-ball accuracy. Original visual anchors remain unchanged.

The consensus override produces **no change** relative to the conservative point hybrid and is not needed for the replay. The two previously reviewed stick-blade errors at frames 4021 and 4026 remain: point-SAM fixes them from pilot seed 3989, but fails from the available nearby seeds in the 31-seed run. This sensitivity is a real limitation, not a solved failure.

The fixed output audit inspected 24 evenly spaced changed SAM positions: **20 visibly within 25 pixels, four unverified, no clear wrong location**. Unverified cases are a ball merging with a blade, a blade-tip blur, a partially occluded hand and a held ball indistinguishable from the official hand. Review uses raw neighboring frames; coordinates are approximate and rounded to five native pixels. This is correlated assistant output review, not an independent test.

The conservative point-SAM + TAP hybrid is the default for this existing replay. The prior method remains selectable. [Open the comparison](https://hockey-tracking-phi.vercel.app/overhead.html?ball=sam&t=124.6667).

## Method

1. Freeze six previously published visual seeds, with three development windows and three validation windows. No reviewed localization coordinates become tracking prompts.
2. Use floor homographies to move each crop with the camera. This compensation is approximate for airborne or held balls. Stop at the known camera cut.
3. Compare a positive point, a 16×16 native-pixel box, and the same box in a tighter crop. Native crops are 768×576 or 384×288, resized by SAM's image encoder. Each seed has up to six seconds before and after it.
4. Propagate SAM's mask memory forward and backward. Convert the largest positive mask component into one candidate center. Compare raw centers, mask-size filtering, and the existing ball-appearance verifier with its frozen 0.3 threshold.
5. Choose the SAM configuration using development windows only: correct points within 25 pixels minus three times the number outside 25 pixels, then points within 12 pixels and fewer errors. Publish the validation-window result separately.
6. Extend the selected configuration to the same 31 visual seeds as the existing TAPNext++ run. Compare a hybrid using the existing context, appearance and continuity gates. On agreement within 25 pixels, keep the higher appearance score. On disagreement, SAM must lead by at least 0.2; otherwise keep TAP. Preserve every original visual anchor.
7. For the exploratory point-prompt candidate, inspect up to 24 evenly spaced added or changed SAM positions using full frames and raw neighboring crops. Invisible or unclear balls do not count as correct.

The separate [fusion amendment](fusion-amendment.json), recorded before reading the 31-seed outputs, also tests a consensus override: two distinct SAM seeds must agree within 12 pixels, both score at least 0.9 on appearance, and the best SAM score must be within 0.05 of TAP. It applies only to conflicts beyond 25 pixels. This targeted development rule addresses an already known TAP drift case; it is not a new independent validation. Both conservative and consensus results are retained.

The product still emits **one ball position per frame**. Masks, shape filters and model scores are intermediate evidence, not extra circles or alternative locations in the replay.

## Reproduction

The standalone Modal entrypoint requires an existing `sport-scans-experiments` volume with the original aligned source frames, cached approved SAM3 checkpoint and private crop-verifier checkpoint. Neither media, weights nor vendor code is included in Git. Pinned revisions, input hashes and all numeric outputs accompany the experiment.

```bash
export MODAL_IMAGE_BUILDER_VERSION=2025.06
modal run -e main experiments/ball-sam-20261026/modal_run.py --task probe
modal run -e main experiments/ball-sam-20261026/modal_run.py --task smoke
modal run -e main experiments/ball-sam-20261026/modal_run.py --task pilot
# Use the configuration selected by evaluate.py for the 31-seed run:
# modal run -e main experiments/ball-sam-20261026/modal_run.py --task all --variants SELECTED_VARIANT
python scripts/verify_ball_sam.py
```

The pinned SAM3 folder loader only recognizes JPEG names. The adapter uses its own decoder/resizer on lossless PNG crops, assigns the same float32 normalized tensor to the tracker state, and preserves native image coordinates. An initial image build used the workspace's legacy builder and failed; a second uncached CPU build was stopped after the original cached image became available under the 2025.06 builder. The first GPU smoke test exposed the filename limitation and was rerun after fixing the adapter. These failures produced no scored predictions.

## Scope

These are approximate assistant-reviewed positions on a previously examined development recording. Seed windows are separated for configuration selection, but this is **not an independent game or independent human ground truth**. Seed frames and their ±3 neighbors are excluded from localization scores. The six-seed and 31-seed experiments have different reference samples. Conditional accuracy among emitted positions does not measure full-video recall or the correctness of invisible-ball estimates.

[SAM3 source](https://github.com/facebookresearch/sam3) · [Previous sequence and point-tracker experiment](../ball-sequence-20261026/)
