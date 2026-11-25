# Temporal chart grounding benchmark

This independently authored benchmark asks which of two candidate answers a fictional record supports for the current encounter. A previous encounter in the same record can contain a plausible competing answer. All facts come from the versioned manifests in this directory. They describe fictional records and require no external downloads.

| Split | Layouts | Fields | Matched groups | Labeled candidates |
| --- | ---: | ---: | ---: | ---: |
| Training | 4 | 6 | 48 | 768 |
| Validation | 3 | 4 | 24 | 384 |
| Locked test | 4 | 5 | 40 | 640 |

Each split uses different layouts, fields, and distinguishing answer terms. Those identifiers and terms are also disjoint from the original 240-example chart dataset. The splits can still share English words and tokenizer subwords. The final test covers both a wording shift and a field-vocabulary shift relative to development.

## How examples are matched

A group fixes one layout, one field, and a pair of candidate values. The generator makes each candidate correct once, presents each chart in both temporal orders, and evaluates both the correct and competing answer. It repeats this crossover under two conditions.

- **Prior decoy.** The competing answer is present in the previous encounter. Exact text matching alone cannot distinguish the two candidates.
- **Absent control.** The previous encounter contains a third value. The competing answer's distinguishing term is absent everywhere in the prompt.

This yields 16 labeled candidates per group. Each `pair_id` identifies an identical chart and question with two alternative completions. A candidate occurs equally often with either label within each group, order, and condition. Several templates interpose a neutral sentence between the visit entries. Some alternatives share wording around their distinguishing term, such as different designs for the same item.

## Reproducing a split

Pass an explicit split to the generator. Development reads only its requested manifest. The generator checks that file against the SHA-256 pinned in [temporal_data.py](../mech_interp/temporal_data.py).

```shell
python -m mech_interp.temporal_data --split train --output runs/temporal-train.jsonl
```

`build_temporal_dataset("train")` and `build_temporal_dataset("validation")` return the same records in memory. Reserve `build_temporal_dataset("test")` for frozen evaluation. The three source files are tracked in Git for reproducibility. Holding out the test records depends on following the evaluation procedure.

Use `prompt + completion` directly, including the leading completion space. Label 0 means supported and 1 means unsupported for the current encounter. `target_start` and `target_end` select the distinguishing term in the completion. `current_start/end` and `prior_start/end` select the full source values in the prompt. All offsets use Python characters and half-open intervals. `chart_order` records the physical order of the visits. `alternative_location` identifies the prior-decoy or absent condition and is shared by both candidates for a chart.

Structured source values, their annotated spans, and the mention-condition flag reveal information beyond the raw chart text. Any method using them during prediction must document those inputs. For uncertainty estimates, keep matched groups together. Groups sharing the same field or layout also share phrasing. This measures discrimination on controlled, teacher-forced candidate answers. Naturally occurring clinical errors require a separate dataset.
