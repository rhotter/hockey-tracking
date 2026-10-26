# Crop-verifier training

These classifiers reject non-ball proposals from the existing NBHL detector.
They do not create ball positions. All task-specific training images are ball
hockey; the generic initialization is ImageNet ResNet18.

| Variant | Positive frames | Negative crops | Best crop validation: correct / wrong / missed |
| --- | ---: | ---: | --- |
| v1: global features | 92 | 475 | 11 / 2 / 5 |
| v2: current-game adaptation | 115 | 624 | 11 / 0 / 5 |
| v3: appearance hard negatives | 115 | 6,144 | 13 / 1 / 3 |
| v4: spatial center features | 115 | 6,144 | 14 / 0 / 2 |

Validation contains 16 old New Jersey positive crops and 113 non-ball crops.
The table reports each checkpoint's selected crop threshold, not its final
tracking rejection threshold. It is a small, reused model-selection set, not a
full-frame or independent tracking benchmark. The four other regression
recordings do not supply training images or checkpoint-selection labels.

The original 92 positives come from the California and Georgia pilot. Adaptation
adds 23 reviewed positive frames from the first 100 seconds of the current New
Jersey replay: eight raw-frame labels and 15 development-audit labels. Those
positions are now training data. Eleven unequivocal development false objects
provide manually reviewed negatives. None of the nine later prototype-audit
errors is used as a negative training crop, although those errors motivated the
model changes and informed the eventual rejection-gate selection.

Automatic negatives are detector proposals far from a known visible ball, plus
random distant centers. Versions 3 and 4 additionally mine warm-colored connected
components in those same positive training frames. This assumes one active ball:
a spare ball or misleading source annotation can introduce label noise. Up to 48
such components per frame are sampled. Negative training batches draw 20% from
reviewed errors, 60% from appearance mining, and 20% from all negatives. Positive
and negative examples are balanced in the minibatches.

Each candidate supplies a 64×64 native crop and a 192×192 context crop, both
resized to 128×128 with reflection padding and ImageNet normalization. Versions
1–3 globally pool the shared encoder's features. Version 4 removes the initial
stride and max-pool, stops after ResNet layer 3, and preserves stride-four feature
maps. A centered 4×4 feature pool and a global pool from each view feed a small
classifier. The center pool preserves information about the proposed point;
global pooling alone can overlook a ball that occupies only a few pixels.

Training uses seed 17, AdamW (learning rate 0.0001, weight decay 0.001), batch 64,
20 epochs, mixed precision, ±3 native-pixel crop jitter, horizontal flips, and
small brightness/contrast changes. The checkpoint maximizes crop-validation F1
over thresholds 0.1–0.9. No calibrated probability claim is made for its scores.

Each folder preserves the epoch history, selected checkpoint hash, numeric sample
coordinates and source-image hashes. Source images and weights stay on Modal.
`crop_verifier.py` and `modal_verifier.py` contain the complete implementation.
