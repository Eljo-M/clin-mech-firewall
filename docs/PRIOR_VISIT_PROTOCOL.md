# Prior-visit experiment protocol

## Question

Can a probe catch an answer copied from an earlier visit while still allowing supported answers on unfamiliar charts? The archived baseline missed all ten prior-visit substitutions at its published threshold. Those ten cases motivated this study. They do not count as fresh test cases.

## Development and freeze

Use the same pinned Qwen checkpoint, raw prompt format, last target-overlapping token, zero-based residual block 12, and logistic scoring interface as the [baseline](RESEARCH.md). Keep the original weights and threshold as a fixed comparison. Compare three candidates before opening the new test split.

1. Recalibrate the original probe’s threshold on the new validation layouts.
2. Fit a new L2 logistic probe on original training rows plus new training rows, with equal row weights.
3. Fit the same model, counting both supported and unsupported members of each prior-decoy pair three times. Both retrained models use L2 strength `0.1`.

For each candidate, choose a threshold whose supported-answer false-positive rate is at most 10% **on every validation layout**. Among feasible thresholds, maximize detection of prior-visit substitutions, then detection of absent substitutions, then minimize total false positives. Prefer the higher threshold in a remaining tie. Select the model using the same detection and false-positive criteria. Break a tie across models in the order listed above. The 10% limit applies to development selection. Report the achieved rate on fresh cases, including any layout where it exceeds 10%.

Use separate fictional source manifests for training, validation and test, with whole matched groups in one split. Hold out both chart layouts and the test fields and their values. Publish the selection file, fitted weights and frozen test-source digest in Git before running the fresh evaluation. Keep the v1 artifacts intact.

## Outcomes

The primary outcome is the difference from the published baseline in the fraction of **prior-visit substitutions caught** on the same fresh rows. Record newly caught and newly missed cases, supported false alarms, and catches on absent substitutions. Break results down by chart layout and the order of the two visits. Include the rethresholded baseline to distinguish a threshold change from a change in what the activations encode. Report the results from every candidate, while identifying the one selected before testing.

The secondary check uses actual greedy generation on the fresh prompts. From the prior-decoy charts, select four distinct prompts per `(layout, chart order)` by the smallest SHA-256 of `boli-generation-v2` followed by a null byte and the UTF-8 prompt. Generate at most 16 tokens each. Score the same tokens with the published and selected probes. Record the first policy crossing and exact released prefix. Classify a generated answer only when it opens with exactly one of the complete current or prior values, ignoring case and an optional opening article. For these directly matched answers, count crossings by the end of the matched value. Report unclassified openings separately and publish the full outputs. This check measures how an answer-trained probe behaves at generation time, including halts on words before the disputed detail.
