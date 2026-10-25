# Roboflow replacement experiment

Actual upstream Roboflow Sports, Trackers and RF-DETR implementations, evaluated on NBHL ball hockey. No task data or specialized weights from ice hockey, soccer or basketball are used. See the [public report](https://hockey-tracking-phi.vercel.app/roboflow.html) for results and uncorrected five-minute replays.

The original experiment compares 20 configurations: cached YOLO detections and an existing NBHL-trained RF-DETR checkpoint, five trackers, and two lost-track buffers. The trackers are Supervision ByteTrack (the Sports tutorial), plus Roboflow Trackers ByteTrack, OC-SORT, C-BIoU and BoT-SORT. The adapter supplies actual frame rate, resets at scene cuts and requires every emitted box to match a current detection. Track IDs remain fragments.

Six Massachusetts development clips contain 3,300 frames. Four 30-second Pittsburgh/Maryland clips and a five-minute Pittsburgh clip contain 6,300 regression frames. For each detector, the objective and tie-break in the protocol select one tracker before new regression scores are computed. The original tutorial comparator is retained. References and recordings were used in earlier campaigns; this is reused sparse assistant-reviewed regression, not independent human ground truth or continuous MOT accuracy.

The RF-DETR medium/960 checkpoint was previously trained on 207 NBHL frames, with 1,531 labeled players and goalies; validation uses 23 frames from another recording. This experiment performs no new weight training. Its weight hash, source revision and detector-output class audit are included. All returned detector proposals enter class-agnostic tracking; raw class IDs remain in the cached detector output. Current RF-DETR and the original tutorial require different Supervision versions, so detection runs in a separate image.

Sports TeamClassifier runs its original SigLIP → UMAP → KMeans pipeline, with deterministic seeds. It adapts on unlabeled automatic detections from each recording. Team reference labels only align cluster names for evaluation. The torso-crop follow-up was declared after seeing the stock classifier fail on Maryland. Both results remain published. The margin/radius gate can confidently accept mistakes.

The hybrid protocol was declared before the new tracking regression scores. It retains the development-selected Roboflow tracker on YOLO boxes, then applies cached generic OSNet body-region features and the unchanged existing global linker (`global-005`). It uses no Qwen image embeddings. This tests replacing SAM2 while retaining custom reconnecting, not an entirely Roboflow pipeline.

Other utility checks run Sports ViewTransformer on the existing manual New Jersey relative calibration and Sports BallTracker on saved NBHL ball proposals. The jersey experiment applies a tutorial-style consensus rule to archived VLM readings; it does not run or validate a Roboflow number detector or classifier. Metric distances and game statistics remain unverified.

## Reproduce

This directory preserves the campaign source layout. The Modal volume `sport-scans-experiments` in environment `main` must already contain the hash-identified input frames, cached detections, appearance vectors, references and checkpoint. Raw recordings, crops, model weights and vendor code are not included. These commands are not a download-and-run demo without those inputs.

From this directory, with an authenticated Modal SDK:

```bash
export MODAL_IMAGE_BUILDER_VERSION=2025.06
modal run -e main scripts/modal_nbhl_roboflow.py --task preflight
modal run -e main scripts/modal_nbhl_roboflow.py --task detect --phase all
modal run -e main scripts/modal_nbhl_roboflow.py --task detector_audit
modal run -e main scripts/modal_nbhl_roboflow.py --task track --phase development
modal run -e main scripts/modal_nbhl_roboflow.py --task select
modal run -e main scripts/modal_nbhl_roboflow.py --task track --phase regression
modal run -e main scripts/modal_nbhl_roboflow.py --task hybrid --phase regression
modal run -e main scripts/modal_nbhl_roboflow.py --task teams
modal run -e main scripts/modal_nbhl_roboflow.py --task teams --clips torso
modal run -e main scripts/modal_nbhl_roboflow.py --task utilities
modal run -e main scripts/modal_nbhl_roboflow.py --task export
```

Top-level packages and upstream source revisions are pinned in the launcher; the preflight records installed versions and upstream implementation hashes. The archived `modal_campaign.py` contains only the shared image/volume definitions, with unrelated campaign entrypoints removed. The existing linker is copied unchanged from the public package. The full RF-DETR image has transitive dependencies resolved by its build, so this is not a complete environment lockfile or a bitwise-reproducibility claim.

Results live under `/data/ball-hockey/reliability-20261018/roboflow-20261025-v1`. The exporter makes a compact archive of reports, sampled predictions, browser overlays and annotated review sheets; it includes no raw video or model weights. The website reuses the existing hash-verified clip video assets.

From the public repository root, `python scripts/verify_roboflow_results.py` independently recounts regression pair outcomes against the unchanged frozen reference, checks selection, replay coordinates, team scoring and visible-ball localization. `python -m unittest discover -s tests -v` checks the local adapter gates and existing reconnecting package. `scripts/verify_roboflow_browser.cjs` checks the rendered report, downloadable evidence, five-minute video, boxes, frame stepping and method switching.

Upstream: [Sports](https://github.com/roboflow/sports), [Trackers](https://github.com/roboflow/trackers), [RF-DETR](https://github.com/roboflow/rf-detr). Their code is installed from the pinned repositories on Modal, not vendored here.
