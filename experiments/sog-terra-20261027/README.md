# Terra 6.1 for NBHL shots on goal

This experiment substitutes `gpt-6.1-terra` into the existing sequence-based shots-on-goal detector. It uses the same eight minutes from New Jersey, Pittsburgh and South Jersey. These are development recordings, not an independent accuracy benchmark.

## Observed high-reasoning results

| Run | Counted SOG | Need review | Cost scope |
| --- | ---: | ---: | --- |
| Published Astra pipeline | 12 | 13 | $54.55 across all original experimental passes |
| Terra scan + its own context | 2 | 15 | $2.08 for eight minutes, or $0.26/min |
| Terra on Astra proposals | 3 | 18 | $2.12 for the 23 contextual reviews only |

All 25 scans, 23 matched reviews and 12 independently proposed reviews completed. Terra's scan proposes 12 review windows versus Astra's 23. Ten of Astra's twelve estimated SOG release frames fall inside Terra's review cores; two fall outside. This is coverage of another model's predictions, not true shot recall.

Both Terra variants propose all three existing New Jersey goals, but only two pass the frozen count gate. The red breakaway goal is proposed without a directly visible prior score change or official signal. More abstention does not prove lower accuracy: the baseline saves are also unverified contextual estimates.

In the independently proposed Terra pass, 12 of the 15 withheld attempts have medium or low model confidence; the remaining three lack the required evidence. The counting rule also withholds contextual saves whenever the model supplies an alternative explanation. For example, the completed maximum-reasoning Pittsburgh review calls the play at 22 seconds a high-confidence save but mentions a conceivable near-post miss returning from the boards. It remains unresolved under the frozen rule. This illustrates an interaction between the model's answer format and the counting policy; relaxing that policy would require a separate, labeled validation.

For the 20 original context requests completed by both models, conservative estimated cost is $19.79 for Astra and $1.78 for Terra, an 11.1-fold reduction. Median response time is 135.7 versus 37.8 seconds. This excludes input preparation/token counting and compares separate runs, not a controlled latency benchmark. The original full Astra campaign includes an earlier strict verifier and recovery; its $6.82/min figure should not be confused with a single production pipeline pass.

## Experimental design

The protocol was frozen before the first inference call. The prompts, sampling and shot-counting rules from the Astra experiment are preserved. The model receives timestamped frames, not a native video stream. The first pass covers every 20-second core with two seconds of overlapping context at 1.875 fps. Context requests use full frames at 2.5 fps, native goal crops at 7.5 fps and up to 30 seconds of sparse aftermath. In both models, scanning uses medium reasoning and context uses high reasoning, with 8192 maximum output tokens.

Three separate phases answer different questions:

- `scan`: Terra independently proposes attempts over all 480 seconds.
- `matched-context`: Terra examines the same 23 proposed windows and native goal crops as Astra. These inputs use the frozen Astra discovery proposals to place the crops, but supply no Astra event classifications. This is a controlled model-substitution diagnostic, not independent discovery.
- `own-context`: Terra examines windows proposed by its own scan, without the intermediate strict verifier. This is the independently discovered Terra pipeline shown in the viewer.

The output `results/own-requests.json` is derived mechanically from Terra scan events. No manual event times, goal references or assistant review labels enter inference. The previous counting policy remains frozen. Original experiments and reference labels remain unchanged. Failures retain their status and reservation; missing answers never mean zero shots. The experiment does not automatically retry ambiguous requests.

## Maximum-reasoning diagnostic

The final diagnostic completed **4 of 9 selected windows**, returning **1 counted SOG and 3 unresolved attempts**. Seven initial requests exhausted their 16,384-token allowance. Recovery with 65,536 tokens completed two of those seven; the other five returned connection failures at approximately the 900-second client timeout, without response usage. Their full reservations remain in the cost estimate, and they were not retried. Missing windows are not zero-shot results.

Maximum-reasoning tests cost an estimated **$3.96**, including **$2.07 reserved for unanswered calls**; the high-reasoning reviews of all nine windows cost $0.94 and all completed. Because completion coverage differs, the returned event counts are not an accuracy comparison. This diagnostic did not establish a reason to replace the high-reasoning pass.

After reading the high-reasoning outputs, `max-protocol.json` freezes a follow-up on all nine original Astra windows containing its twelve estimated SOG. It changes reasoning from high to max and increases the output allowance from 8192 to 16384 tokens, retaining identical prompts and encoded images. This is selected development analysis, not independent discovery or a full-game max pipeline. No high-reasoning answer is replaced. Known responses that exhausted the 16,384-token allowance can receive the separately frozen `max-recovery-protocol.json`: 65,536 output tokens and a 900-second response timeout, with the same max reasoning, images and prompts. Only responses with reported usage and explicit `max_output_tokens` status are eligible; ambiguous failures are never retried. Both passes count against the same campaign budget. The runner waits for all original phases to finish and counts their costs against the same $12 campaign cap. The viewer labels this mode as selected plays.

```sh
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run --detach experiments/sog-terra-20261027/modal_max.py --env-file YOUR_ENV_FILE
```

## Costs and comparison

The finished campaign contains 76 inference requests and has a conservative API estimate of **$8.16**, including the $2.07 in unresolved reservations above. The API-reported usage from answered calls prices to $6.08. The independent high-reasoning pipeline is the separate $2.08 subtotal, not the whole research campaign.

The API budget is $12 for the entire campaign, including the additional maximum-reasoning diagnostic above. Independent pipeline cost comprises `scan` plus `own-context`; the separate matched-context diagnostic is additional evaluation expense. Standard-rate estimates use API-reported ordinary input, cache-read, cache-write and output tokens. The rates checked on 2026-10-27 are $1 for ordinary input, $0.05 for reads, $1.25 for default 30-minute writes and $5 for output per million tokens. The report recomputes these disjoint categories; the raw ledger `standard_rate_usd` field is a preserved legacy base-rate estimate that omits cache-write premiums, not billed cost. A separate conservative estimate adds the same 25% input-rate margin used in the prior reports and retains unresolved reservations. Modal compute/storage is excluded. These estimates are not invoices.

The recovery runner is `modal_max_recovery.py`; it must run after the initial maximum-reasoning phase finishes.

Pricing sources: [OpenAI release announcement](https://openai.com/index/introducing-gpt-6-1-terra/), [model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-terra), and [cache pricing rules](https://developers.openai.com/api/docs/guides/prompt-caching).

Predictions are aligned one-to-one within the same recording and team, with a frozen two-second release-time tolerance. Alignment maximizes match count then minimizes total timing error; count category and outcome do not influence pairing. Agreement with Astra is not accuracy. The three existing New Jersey goal references are checked only after inference, as a limited sanity check. An additional diagnostic separates proposed goal outcomes from goals that pass the frozen evidence gate; a proposed goal is not automatically counted. They do not validate saves, shot recall or shooter attribution.

## Reproduction

```sh
python scripts/analyze_sog_terra.py
python scripts/verify_sog_terra.py
python -m unittest discover -s tests -p test_sog_comparison.py -v
```

The runner uses the existing private 15 fps frame corpus in the Modal volume. Supply your authorized key through `OPENAI_API_KEY` or an environment file; it is mounted as a transient Modal secret, never written into results.

```sh
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run --detach experiments/sog-terra-20261027/modal_run.py --env-file YOUR_ENV_FILE
```

Use a separate experiment/output path for intentional reruns. Existing request IDs are not silently issued again. Source-frame hashes, encoded-image hashes, token usage, response status, model identity and timing are retained in the ledgers. Raw videos and input images are excluded from Git.
