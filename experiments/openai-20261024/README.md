# OpenAI ball-hockey experiments — October 24, 2026

Both experiments finished. The existing SAM2 + OSNet + joint Qwen2B pipeline is retained: these results do not establish a full-game replacement.

## Direct sampled-video baseline

GPT-6 Astra and GPT-6.1 Sol each received three independent full-task requests: Pittsburgh 10:00–10:30, Maryland women 10:00–10:30, and Pittsburgh 15:00–20:00. Only chronological full frames, timestamps and task instructions were supplied. The models received no detector output, crops, tracks, labels or roster. This is sampled video as images, without native video or audio: 1 fps on short clips, 0.5 fps on the long clip, plus each final frame (213 images per model). Long frames were resized to width 1280. Output boxes were requested every five or thirty seconds plus the final frame.

Each model matched 96/96 reference appearances and 69/69 same-player checks on the two short clips, with zero joins among 609 different-player checks and zero unverified joins. The existing pipeline matched 94/96 appearances and 65/69 same checks, with zero known wrong joins and two unverified joins. Each model's five-minute request reached the 24,576-token output limit and returned no completed structured answer. These two failures are retained separately, not presented as ordinary zero-quality tracking predictions. Four of six requests completed.

The reference was frozen before this experiment and not supplied to inference. It is sparse, reused and assistant-reviewed, with no independent human verification. The 69 checks share observations in fourteen sampled frames; they are not independent identities or full-game accuracy. The pipeline used 15 fps, so this is an engineering baseline with different temporal information. References are not exhaustive: detection precision is unknown. Pair matches use one-to-one IoU ≥ 0.5; missing observations are not correct rejections and duplicate IDs within a frame cannot score.

All emitted floor times, jersey numbers, shots, possession, scoreboard and event assertions remain model predictions. None establishes independently verified game-stat accuracy. Both models left metric distance unknown. Some short-clip floor times and shot hypotheses were populated; they are not published as measured player totals. No long-clip statistics were recovered from the incomplete answers.

## Reconnection pilot

The same original crops and prompt as archived Qwen evaluations were used for 65 hash-selected candidate links across nine reused ball-hockey development clips, in both image-group orders. Candidate selection did not use identity labels or model outcomes. The protocol shared a $40 cap; the direct-video experiment had a separate $10 cap. No runner retry was made for an already dispatched request.

| Model | Valid / 130 | Order agreement | Same/different flips | Stable decisive / 65 | Forward uncertain / 65 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 129 | 51/64 | 1 | 34 | 26 |
| GPT-6.1 Sol | 130 | 50/65 | 2 | 32 | 25 |
| Qwen3-VL 8B | 130 | 51/65 | 7 | 42 | 15 |
| Qwen3-VL 235B | 130 | 51/65 | 7 | 46 | 9 |

One Astra request remains a reservation without a saved response or usage. Its outcome and billing are unknown, and its maximum reserved cost is retained. OpenAI makes fewer direct contradictions while abstaining more; order agreement is similar and is not identity accuracy. No OpenAI reconnection policy was applied to the full tracker in this pilot.

Post-run visual review examined all three OpenAI same/different flips against the original crops. The first Astra case compares lower-leg markings with different visible portions across front/back views. One Sol case includes other people within the crop group; the final case has a clear 76 in one group but no complete readable number in the other, despite the same-player answer claiming both show 76. These observations diagnose unreliable evidence claims; they do not introduce new ground-truth labels.

Conservative API estimate: $8.84836 reconnecting + $7.63875 direct video = $16.48711. Includes one unresolved reservation and 1.25× uncached input price to cover cache writes; ignores cache discounts and excludes Modal charges. It is not an invoice. Saved usage and elapsed generation times are included.

## Reproduce saved measurements

From the repository root, with project Python dependencies installed:

```sh
python scripts/verify_openai_results.py
python -m unittest discover -s experiments/openai-20261024 -p 'test_*.py' -v
```

The verifier checks input hashes and recomputes both reports from saved manifests, ledgers, Qwen answers, unchanged reference labels and existing pipeline predictions. It performs no API calls. Inputs and reports are in `web/data/reliability/openai-20261024/`. Incomplete long requests produce missing pairs in the full machine report; the website separately reports the completed short subset and the long failures.

Archived runner and analyzer source is preserved here; Modal wrappers expect the original laboratory layout and persistent input volume. Image manifests contain hashes, but raw frames/crops, credentials, API request image payloads and model weights are not included. Reproducing fresh inference requires these external inputs and an authorized API account. No inference is needed to audit saved results.
