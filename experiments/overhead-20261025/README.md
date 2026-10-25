# Five-minute NBHL overhead replay

[Public replay](https://hockey-tracking-phi.vercel.app/overhead.html). This processes recording 15:00–20:00 of [Bellmawr Hawks vs. South Jersey Nights Watch, August 30, 2026](https://www.youtube.com/watch?v=OLpK8enSjJs&t=900), at 15 fps: 4,500 frames, including stoppages, pans, occlusions and an edited transition. It is a development example from a previously used game, not a new-game benchmark.

## Selected player

Red jersey **#50** has 72.8 seconds of accepted floor positions in this 300-second recording interval. Three SAM2 fragments are joined by two explicit assistant review decisions. The review contains 123 one-second, endpoint and gap-boundary crops; readable jersey numbers support the joins. A crowded ending is excluded. Other white-helmet players are not merged based on appearance alone. Missing time is unknown, not proven bench time. This is a trajectory example with partial coverage, not five uninterrupted minutes of verified tracking.

## Pipeline

1. Extract all frames from the source and encode the website video from those same frames. This keeps annotations and playback synchronized.
2. Run the existing NBHL-finetuned YOLO11m person detector and a separate YOLO11m pose model. Pose detections are associated one-to-one by bounding-box overlap.
3. Propagate SAM2.1 large masks on an H100. Associate each mask with a current detector box, require an overlap margin against competing players, retire stale tracks, and reset on the black transition. A current detection is required to emit a box. No automatic reconnection model is used in this experiment.
4. Classify track fragments using the existing frozen DINOv2 body/torso/legs role classifier. Median fragment scores filter likely spectators and officials; these scores do not establish jersey identity.
5. Register the moving camera against 32 reference frames from this recording. SIFT matching, robust homography fitting, feature spread, residual and reference-agreement gates operate every five frames. Compare all-background matching against floor-only reference features. A manual floor calibration supplies the reference coordinate system. This still requires venue/reference setup; it is not a universal automatic rink-landmark detector.
6. Fit a global floor rectification using paint/board marks on three frames. Compare alignment on seven other marked frames. Interpolate only between accepted adjacent camera keyframes with bounded changes. Explicitly withhold coordinates through the edited transition.
7. Use the midpoint of two confident pose ankles, with truncation and mask-bound checks, as the estimated floor contact point. Transform it into relative rink coordinates. Hidden feet have no fallback. The current pose need not be perfect even when these gates pass. Ankles sit above the sole contact point, so this approximation can also bias the projected location; toe/sole localization and independent floor-point labels remain needed for distance statistics.
8. Keep the selected player's explicit assistant-reviewed identity associations separate from automatic track IDs. Break the trajectory across missing frames, identity-fragment boundaries, invalid calibration and abrupt displacements. The map never fills off-camera gaps.
9. Run the existing ball detector and emit sparse floor-projected candidates after confidence, ambiguity and temporal-support gates. This is not a validated complete ball trajectory; airborne-ball height cannot be recovered by the floor homography.

## Geometry check and its limits

The original floor reference contains 69 manually marked points on ten frames. `floor-line-review-v1.json` is preserved. Some approximate blue-line marks were misplaced; `floor-line-review-v2.json` refines those marks using the original blue paint pixels, with no camera estimates read by the refinement script. This happened after seeing initial aggregate errors, so the check is **not blind or independent**.

The global correction uses frames 450, 1800 and 2700, with 20 points. One occluded center-line point on a referee's body is excluded. The seven remaining frames supply 48 checks. Six points fall on a blurred frame whose camera mapping is withheld; they remain in the denominator.

| Method | Answered / marked | Median image-to-line error | 90th percentile |
|---|---:|---:|---:|
| Original background registration | 42 / 48 | 4.55 px | 18.34 px |
| Floor-only reference features | 36 / 48 | 4.43 px | 17.82 px |
| Background + global floor correction | 42 / 48 | 1.78 px | 4.58 px |

All 42 answered checks after correction are within 10 pixels. These are perpendicular distances to painted/board lines in the 1920×1080 source. They do **not** measure absolute floor-position error, identity accuracy or generalization. Rink dimensions were not surveyed. The layout assumes parallel blue lines equally spaced around center, a straight far board edge and approximate end extents. Relative rink coordinates therefore cannot be reported as miles, speed or true time on floor.

The assistant also inspected overlays every 15 seconds and the final frame, and inspected both sides of the black transition. These sparse reviews are useful diagnostics, not exhaustive human ground truth.

## Reproduce and inspect

The experiment runs on Modal using the existing private source video, weights and camera-bank caches. Those inputs are referenced by SHA-256 in the published reports; they are not bundled into Git. No new detector or SAM2 fine-tuning occurred here.

```bash
modal run -e main experiments/overhead-20261025/scripts/modal_overhead.py --task prepare
modal run -e main experiments/overhead-20261025/scripts/modal_overhead.py --task detect --use-gpu
modal run -e main experiments/overhead-20261025/scripts/modal_overhead.py --task register
modal run -e main experiments/overhead-20261025/scripts/modal_overhead.py --task refine
modal run -e main experiments/overhead-20261025/scripts/modal_overhead.py --task sam
modal run -e main experiments/overhead-20261025/scripts/modal_overhead.py --task roles
modal run -e main experiments/overhead-20261025/scripts/modal_overhead.py --task ball --use-gpu
modal run -e main experiments/overhead-20261025/scripts/modal_overhead.py --task review_tracks
# Inspect identity crops and write explicit player-review.json associations before export.
modal run -e main experiments/overhead-20261025/scripts/modal_overhead.py --task export
python scripts/verify_overhead.py
```

The published JSON, full five-minute video, identity evidence, static trajectory image and CSV are checksum-pinned. The CSV has one row per frame, including explicit missing-position rows. The website shows the camera view, observed boxes and feet, a floor-alignment overlay, overhead positions, a time-colored path, a coverage timeline and player evidence. Automatic fragments remain labeled as fragments.
