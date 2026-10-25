# Hockey tracking

Player-track reconnection experiments on **NBHL ball hockey**. This release contains frozen inference source, baseline and alternative results, auditable pair outcomes, and new-recording evaluations.

The previously highlighted candidate combines **SAM2 track fragments**, **OSNet appearance features**, and **Qwen3-VL Embedding 2B**. On two new recordings, reconnecting added eight correct pair connections but introduced six wrong pair joins. **SAM2 without reconnecting was safer on these new sparse checks.** The reconnecting component remains experimental.

## Website

[Open the public website](https://hockey-tracking-phi.vercel.app).

The website is a **Next.js 16 / React 19** app deployed on **Vercel**, connected to this repository's `main` branch. Pull requests receive Vercel previews. GitHub Actions verifies the research release and Next.js production build.

```bash
npm ci
npm run prepare:site
npm run dev
# Production build:
npm run build
npm start
```

Python 3.10+ and Node.js 22+ are required. The App Router renders the research pages from trusted templates in `web/`, with server-rendered React content and the existing video/canvas controllers initialized after hydration. Existing `.html` links still work. `app/`, `components/` and `lib/` contain the Next.js integration.

Replay assets are stored separately in the versioned GitHub release, pinned by archive and individual-file SHA-256 hashes in `site-assets.json`. The build downloads and verifies them, then prepares generated `public/` assets. To update media, upload a new versioned bundle and update the manifest; keep media out of Git history. Never edit generated `public/` or `.next/` files.

## New recording test — October 25, 2026

[New-game results, reference photos and replays](https://hockey-tracking-phi.vercel.app/new-games.html) · [Experiment source](experiments/new-games-20261025/)

Two unused official NBHL recordings from Pittsburgh and South Jersey were tested with frozen weights and settings. Four fixed 90-second intervals give six minutes / 5,400 frames. The reference contains 177 selected appearances and 145 known same-player connections, annotated from raw frames before tracking inference. There is no independent human verification, and these sparse checks do not establish full-game accuracy.

| Method | Correct / 145 | Wrong pair joins | Unverified joins |
| --- | ---: | ---: | ---: |
| SAM2 + OSNet + Qwen 2B | 116 | 6 | 88 |
| RF-DETR + Roboflow BoT-SORT | 96 | 7 | 16 |
| YOLO + Roboflow BoT-SORT | 98 | 5 | 11 |
| SAM2 without reconnecting | 108 | 0 | 23 |

Reconnecting preserved every box but added 65 unverified joins. Seven remaining positive checks lack a matched endpoint, and 22 have different IDs. The six wrong pair joins involve teammates. A separate review found repeated scene resets during continuous close-view play at Pittsburgh 10:21–10:22, preventing consideration of seven known same-player pairs across that boundary. No settings or reference labels were changed after inference.

## Roboflow replacement test — October 25, 2026

[Results, reference images and five-minute replays](https://hockey-tracking-phi.vercel.app/roboflow.html) · [Experiment source](experiments/roboflow-20261025/)

Twenty configurations compared the actual Roboflow Sports/Trackers implementations with cached YOLO and our existing NBHL-trained RF-DETR detector. Development selected BoT-SORT for both detectors. Across four short clips and a five-minute clip, RF-DETR + Roboflow BoT-SORT recovered 72/129 reviewed same-player connections with three wrong merges. The YOLO/Roboflow hybrid with existing OSNet reconnecting reached 94/129 but made 14 wrong merges. On that reused reference, SAM2 + OSNet + Qwen was preferred at 111/129 with zero known wrong merges; the new-recording results above expose additional failures.

Sports TeamClassifier was useful on Pittsburgh but unreliable on Maryland women. A torso-crop follow-up reached 146/147 correct matched Pittsburgh observations and 43/75 on Maryland. ViewTransformer reproduced the existing relative-coordinate projection; calibration and metric distances remain external. BallTracker added no measured localization benefit. The jersey consensus check did not run a new Roboflow OCR model. These are reused sparse assistant-reviewed checks, with no independent human verification. Recount the published evidence with `python scripts/verify_roboflow_results.py`.

## OpenAI baseline and reconnection pilot — October 24, 2026

[Completed results and limits](https://hockey-tracking-phi.vercel.app/vlm.html#openai-results) · [Experiment source](experiments/openai-20261024/)

GPT-6 Astra and GPT-6.1 Sol each matched all 69 selected same-player checks and 96 reference appearances across two 30-second clips, compared with 65 checks and 94 appearances for the existing pipeline. Both five-minute requests hit the 24,576-token output limit without a completed structured answer. This was timestamped full-frame sampling, not native video, and the reference is sparse and assistant-reviewed without independent human verification. Game statistics remain unverified.

On 65 matched crop reconnection candidates, reversing image order caused one direct same/different flip for Astra and two for Sol, versus seven each for the archived Qwen models. OpenAI abstained more; overall agreement was similar. This does not demonstrate a tracking accuracy gain. That campaign retained the existing pipeline. Conservative API cost for both experiments was $16.49, excluding Modal. Recompute saved reports with `python scripts/verify_openai_results.py`.

## SAM3 fine-tuning and longer replays — October 24, 2026

[Results and evidence](https://hockey-tracking-phi.vercel.app/sam3-improvements.html) · [Complete 20m 15s replay](https://hockey-tracking-phi.vercel.app/tracking-replay.html?clip=pittsburgh-sam3-20min&method=sam3-long)

Six SAM3 weight-training runs used 196 reviewed NBHL frames. The validation-selected detector improved from 52.8% to 88.1% F1 on 36 reused regression frames. A separately selected, post-test rink-filter follow-up reached 90.8%, compared with the existing YOLO detector’s 89.0%. On eleven newly annotated Pittsburgh frames, the frozen fine-tune plus filter reached 96.1% F1; this is a small correlated sample, assisted by YOLO proposals, from a recording previously used for tracking.

That campaign retained the prior SAM2 + OSNet + Qwen tracking baseline. Adding SAM3 agreement retained 111/129 correct sparse connections and zero known wrong merges, while reducing unverified joins from four to three. This is not a verified accuracy gain. Decoder-only SAM3 adaptation also reached 111 correct connections, but introduced three wrong merges. All selected methods, unsuccessful alternatives, and separate short/five-minute counts are published.

The full replay contains 18,225 frames with native SAM3 and two existing SAM2 appearance variants. A separately sealed 31-sample interval matched all 88 visible reference observations with SAM3 and kept the same ID across all 78 scored adjacent transitions. It covers only four initially visible players and does not validate full-game identity. Each 315-second SAM3 run took 65–110 minutes on an H100 and used 223–341 GiB of process RAM; this is an offline research pipeline.

## SAM3 experiments — October 23, 2026

[Results, eight interactive replays and all 48 pose-audit crops](https://hockey-tracking-phi.vercel.app/sam3.html).

Official pretrained SAM3, SAM 3.1 Object Multiplex, a detector-supported SAM3 hybrid, three detection prompts, and SAM 3D Body were tested on Modal. No new fine-tuning was performed. Native SAM3 improved over raw SAM2 from 80 to 90 correct short-clip pairs, but introduced three wrong joins; the existing full pipeline already achieves 90 with zero reviewed wrong joins. Native SAM 3.1's tested configuration scored 41/103 with one wrong join and substantial bounding-box failures. A mask-cleanup follow-up increased correct development pairs from 33 to 37 but added a wrong join, failing the predeclared conservative objective. That campaign retained the existing tracking pipeline.

The standalone SAM3 detector scored 53.3% F1 versus 89.0% for the fine-tuned YOLO detector on the 36-frame regression check. SAM 3D Body emitted ankles on all 48 cases, including all seven with occluded feet; that does not establish coordinate accuracy or visibility. These are reused, sparse checks, not full-game reliability measurements. Experiment source and subsequent capacity/mask-postprocessing diagnostics are archived under `experiments/sam3-20261023/`.

## Measured result

| Method | Four short clips: correct / 103 | Wrong / 962 | Unverified joins | Five-minute clip: correct / 26 | Wrong / 368 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OSNet + global linking baseline | 90 | 0 | 2 | 18 | 2 |
| **OSNet + joint-image 2B embeddings** | **90** | **0** | **4** | **21** | **0** |
| Averaged-image 2B embeddings, primary development selection | 89 | 0 | 2 | 18 | 0 |
| 235B language-model multiple-image veto | 87 | 0 | 1 | 20 | 0 |

The joint-image candidate added three correct five-minute connections and removed both reviewed wrong joins, losing no existing correct pair. It retained the same 90 correct short-clip pairs but added **two unverified joins**. All four rows have zero unverified joins in the five-minute check.

**These are sparse, correlated, assistant-reviewed pair checks on previously evaluated recordings. They are not full-game identity accuracy or independently validated ground truth.** The 2B joint-image configuration was a challenger fixed before regression scoring. It tied the best development objective; the preset tie-break chose averaged-image embeddings instead. We highlight joint-image embeddings after observing regression results.

Nine development clips (320 seconds) and five regression clips (420 seconds) cover four recordings. Four pretrained image-model checkpoints were compared, including 234 embedding settings on development. No new model fine-tuning was performed in this reconnection campaign. The upstream detector had previously been fine-tuned on ball-hockey images.

## Verify the published results

No account, GPU, model download or video is needed. Python 3.10+ is sufficient:

```bash
git clone https://github.com/rhotter/hockey-tracking.git
cd hockey-tracking
python -m pip install .
python scripts/verify_results.py
python scripts/verify_sam3_results.py
python scripts/verify_sam3_improvements.py
python scripts/verify_openai_results.py
python scripts/verify_roboflow_results.py
python scripts/verify_new_games.py
python scripts/verify_release.py
```

The first command recomputes **117 comparisons** from the included pair outcomes: 78 generative-model, 24 embedding, and 15 revised-prompt comparisons. It also checks frozen source, selection hashes and the fixed-observation ceiling. The SAM3 verifier checks saved pair outcomes against the frozen reference, all exported replay frames, development selection and pose-emission counts. The final command checks the public release's file hashes. These checks reproduce saved evaluation counts, **not fresh inference**.

## Use the reconnection component

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/reconnect.py
python -m unittest discover -s tests -v
```

The example uses explicitly synthetic data. `hockey_tracking.reconnect.reconnect()` takes observed tracks, fragment intervals and precomputed candidate similarities. It applies the fixed 50/50 OSNet/Qwen blend and original global assignment. It requires 15 fps and keeps gaps unobserved. See [the input contract](docs/pipeline.md).

**This is a research release, not a turnkey video-to-statistics application.** Full video inference requires upstream detections, model weights, sampled crops and cached features that are not distributed here. The detector's trained checkpoint and the original Modal caches are not public artifacts in this repository.

## Contents

- `hockey_tracking/`: portable reconnection component, extracted from the measured implementation.
- `results/`: pair outcomes, all comparison summaries, exact changed pairs and recording references.
- `experiments/reference/`: selected original source and frozen protocols, preserved byte-for-byte for auditing.
- `experiments/sam3-20261023/`: SAM3, SAM 3.1 and SAM 3D Body experiment source and protocols; outputs are in `web/data/reliability/sam3/`.
- `experiments/sam3-20261024/`: SAM3 weight adaptation, hybrid tracking, visual prompts, longer replays and independent result-recounting code.
- `experiments/roboflow-20261025/`: Roboflow detector/tracker alternatives, team clustering, retained-appearance hybrid, utility checks and pinned protocols.
- `experiments/new-games-20261025/`: frozen-model testing on two unused NBHL recordings, proposal-free reference annotations, Modal runners and reproducible comparison results.
- `experiments/provenance.json`: original source hashes and release origin.
- `docs/`: pipeline, evaluation limits and external artifact requirements.
- `tests/`: assignment, observation-preservation and original-solver equivalence checks.

## Limits and next work

- IDs are automatic fragments, not verified roster identities. Time on floor, distance, shots and possession are not established by these results.
- Direct reconnection candidates span at most 20 seconds and do not cross scene boundaries. Long bench absences remain unresolved.
- In the previous short-clip reference, ten of the 13 missed same-player checks lack a matched box at an endpoint. Reconnection cannot fix those missing observations.
- Similar uniforms, hidden numbers and identity switches inside fragments remain difficult. A language model's confident explanation is not ground truth.
- Earlier recordings were reused for evaluation. The new-recording comparison adds sparse assistant labels on two unused games. Continuous, independently reviewed identity annotation is still needed.

Priorities: recover missing observations, detect and split contaminated fragments, improve crop selection and jersey evidence, then validate persistent identities across shifts and camera cuts.

## Licensing and media

Original project code and included original annotations are released under the [MIT license](LICENSE). Git history excludes video, player crop images, model weights and vendored third-party implementations. The website uses replay clips and player images distributed separately as release assets; those third-party media are excluded from the MIT license. Referenced models and dependencies retain their own licenses and access conditions; this license does not grant rights to those artifacts. See [third-party references](docs/artifacts.md).
