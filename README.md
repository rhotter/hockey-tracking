# Hockey tracking

Player-track reconnection experiments on **NBHL ball hockey**. This release contains the best observed candidate, baseline and alternative results, auditable pair outcomes, and the relevant research source.

The strongest observed candidate combines **SAM2 track fragments**, **OSNet appearance features**, and **Qwen3-VL Embedding 2B**, encoding several player crops together. A global assignment joins compatible fragments without filling missing positions.

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

Python 3.10+ and Node.js 22+ are required. The App Router renders the five research pages from trusted templates in `web/`, with server-rendered React content and the existing video/canvas controllers initialized after hydration. Existing `.html` links still work. `app/`, `components/` and `lib/` contain the Next.js integration.

Replay assets are stored separately in the `site-assets-v1` GitHub release, pinned by archive and individual-file SHA-256 hashes in `site-assets.json`. The build downloads and verifies them, then prepares generated `public/` assets. To update media, upload a new versioned bundle and update the manifest; keep media out of Git history. Never edit generated `public/` or `.next/` files.

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
python scripts/verify_results.py
python scripts/verify_release.py
```

The first command recomputes **117 comparisons** from the included pair outcomes: 78 generative-model, 24 embedding, and 15 revised-prompt comparisons. It also checks frozen source, selection hashes and the fixed-observation ceiling. The second checks the public release's file hashes. These checks reproduce saved evaluation counts, **not fresh inference**.

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
- `experiments/provenance.json`: original source hashes and release origin.
- `docs/`: pipeline, evaluation limits and external artifact requirements.
- `tests/`: assignment, observation-preservation and original-solver equivalence checks.

## Limits and next work

- IDs are automatic fragments, not verified roster identities. Time on floor, distance, shots and possession are not established by these results.
- Direct reconnection candidates span at most 20 seconds and do not cross scene boundaries. Long bench absences remain unresolved.
- Ten of the 13 missed short-clip same-player checks lack a matched box at an endpoint. Reconnection cannot fix those missing observations.
- Similar uniforms, hidden numbers and identity switches inside fragments remain difficult. A language model's confident explanation is not ground truth.
- Current recordings have been reused for evaluation. The next meaningful test is continuous, independently reviewed identity annotation on new NBHL recordings.

Priorities: recover missing observations, detect and split contaminated fragments, improve crop selection and jersey evidence, then validate persistent identities across shifts and camera cuts.

## Licensing and media

Original project code and included original annotations are released under the [MIT license](LICENSE). Git history excludes video, player crop images, model weights and vendored third-party implementations. The website uses replay clips and player images distributed separately as release assets; those third-party media are excluded from the MIT license. Referenced models and dependencies retain their own licenses and access conditions; this license does not grant rights to those artifacts. See [third-party references](docs/artifacts.md).
