# Shots on goal from NBHL ball-hockey video

This adds automatic **event detection**, separate from the ball-position tracker. The public viewer at [shots.html](https://hockey-tracking-phi.vercel.app/shots.html) provides estimated team SOG, release times, original recording links, slow-motion replays and unresolved attempts. Numbers are segment estimates, not official or full-game statistics. Player attribution is unverified.

## Observed outputs

| Segment | Estimated SOG | Goals included | Unresolved attempts |
| --- | ---: | ---: | ---: |
| New Jersey, 300 seconds | 8 | 3 | 8 |
| Pittsburgh finals, 90 seconds | 2 | 1 | 3 |
| South Jersey finals, 90 seconds | 2 | 0 | 2 |
| Total | 12 | 4 | 13 |

All 12 counted events use contextual inference. These are model outputs, not independently measured correct-event counts. The New Jersey goal detections overlap all three pre-existing official-goal/reviewed-video intervals within a two-second matching tolerance; that sanity check does not validate saves or SOG recall.

The scan completed 25/25 windows; strict verification completed 22/23, with one connection failure. The context pass completed 20/23, with two known incomplete responses and one connection failure. A separate medium-reasoning pass completed both known-incomplete sequences. One opening Pittsburgh sequence uses the completed strict verifier as fallback. No proposal remains entirely unprocessed. The initial strict verifier counted zero SOG; context reasoning makes event estimates possible, but this is not a measured accuracy comparison.

Conservative OpenAI API cost including unresolved failure reservations: **$54.55**, below the $60 cap. Modal compute/storage is additional and excluded from that estimate. Full raw answers, usage and failure states are retained in the four phase ledgers.

## Method

1. GPT-6-astra scans the full segments in disjoint 20-second cores with two seconds of overlapping context, sampled at 1.875 frames/s and resized to 960px wide. This proposes shooting sequences, not SOG counts.
2. Automatically merge nearby proposal intervals. A separate request examines each interval at 5 frames/s and 1280px width. Its actual settings are in `verify-requests.json`; they override the initial protocol's denser defaults. The verifier sees raw images without the discovery answers.
3. A context experiment adds 960px native goal-mouth crops at 7.5 frames/s, full views at 2.5 frames/s, and sparse aftermath extending up to 30 seconds. Crop centers come from scan proposals. Context ledger records retain the seed request's `stride` and `width`; the actual context samples are listed in `full_frames`, `crop_frames` and `sample_frames`, with settings in the separate context protocol. This pass can infer saves from the shooting action, keeper interception and a keeper-origin rebound or goalward path fragments. Delayed scoreboard changes, referee signals and center restarts help identify goals with an obscured crossing. It distinguishes direct evidence from contextual inference and reports alternative explanations.
4. `hockey_tracking/shots.py` applies explicit count rules, preserves distinct rebound releases, and merges only compatible cross-window duplicates. Misses, skater blocks, post-only shots and disallowed goals are excluded. Ambiguous attempts stay in review. A completed context answer replaces the stricter verifier; failed context calls fall back to a completed verifier. Missing results are recorded, never treated as proof of zero shots.
5. `scripts/detect_shots.py` produces the JSON, CSV and report displayed by the Next.js page. It requires no credentials or inference to reproduce published decisions.

The initial pixel-confirmation approach abstained even on scoring plays, motivating the context pass. This is a development experiment. The counting rule was revised after the first contextual save sequence: a goalkeeper-origin rebound is useful evidence even when play continues without a whistle. A subsequent development correction accepts a defensive-zone restart as corroboration of a held/gloved save, provided a clear shooting action, keeper interception, high confidence and no alternative are all present. The earlier rule systematically excluded saves without a rebound. All three policy hashes and the decisions are recorded in `count-policy.json`. No claim of a fresh independent accuracy improvement is made.

## Evidence and limitations

All 480 seconds were inspected by the assistant on chronological raw contact sheets before reading event classifications (proposal times were known). `raw-review-notes.json` records that review; it is **not independent human ground truth**. Larger goal-mouth sequences were subsequently inspected. Precision, recall and full-game accuracy have not been measured. Scan coverage means every interval received a scan, not that every shot was found.

These recordings contain tiny, blurred and occluded balls. Contextual calls can mistake a pass or wide attempt for a save. Confidence tiers are model judgments, not calibrated probabilities. Shooter numbers are optional visual readings and are not linked to persistent verified player identities. Release times are approximate recording timestamps, not game-clock times. The Pittsburgh light-gray team retains the original `black` compatibility code, corrected in display labels.

`results/*-ledger.json` preserve raw structured answers, request outcomes, source frame hashes, token usage and conservative cost estimates. No reference labels were supplied to inference. The raw review images and videos are excluded from Git. The existing checksum-pinned public replay archive supplies the site videos.

The initial scan/verification budget was $40. The explicitly versioned context protocol raises the **whole campaign** limit to $60, including those earlier calls. Ambiguous failures keep their reserved cost and are not retried automatically. OpenAI API cost estimates ignore cache discounts, exclude Modal compute/storage, and are not invoices. The first verification pass uses a 180-second client timeout; the context experiment allows 360 seconds. Known incomplete context responses can receive a separate medium-reasoning pass with a larger output allowance under the same total budget. This pass never reissues ambiguous connection failures; its raw ledger remains separate.

## Reproduce

```sh
python scripts/detect_shots.py
python scripts/verify_sog_results.py
python -m unittest discover -s tests -p test_shots.py -v
npm run build
```

For another recording, extract timestamp-aligned 15fps frames into the configured Modal volume, create a separate protocol and presentation metadata, and run the discovery/verification/context stages. This experiment uses an existing private frame corpus on Modal; the inference inputs are not bundled in Git. Supply your own authorized OpenAI key via `OPENAI_API_KEY` or an environment file. The runner mounts it as a transient Modal secret and never writes it into result artifacts.

```sh
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run --detach experiments/sog-20261027/modal_run.py --task scan --env-file YOUR_ENV_FILE
python scripts/propose_shot_windows.py --output verify-requests.json
# Upload this file to the experiment folder on the Modal volume before verify.
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run --detach experiments/sog-20261027/modal_run.py --task verify --env-file YOUR_ENV_FILE
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run --detach experiments/sog-20261027/modal_run.py --task context --env-file YOUR_ENV_FILE
```

Existing request IDs are never silently reissued. Use a new experiment/output directory for an intentional rerun. The scan worker's original hash is recorded separately because diagnostic persistence was improved before verification.

Browser regression checks are in `scripts/check_shots_page.cjs` (optional Playwright runtime). Set `SOG_TEST_BASE` to a running local or public site, and `CHROME_BIN` if using an installed Chrome. The check loads all three videos, selects attempts, steps frames, checks deep links and inspects the mobile layout.
