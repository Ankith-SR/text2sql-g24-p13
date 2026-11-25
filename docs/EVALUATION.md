# Evaluation protocol

Every system is evaluated on the same dev set, the same prompt format and the same database
files (proposal, Section V). All numbers come from `scripts/run_eval.py`.

## Systems
| Tier | Command | Run name |
|---|---|---|
| 1 Heuristic | `python scripts/run_eval.py --system heuristic --dataset spider` | `heuristic-spider-none` |
| 2 LGESQL | train in its own repo (`docs/LGESQL.md`), then `python scripts/score_preds.py --preds <file> --name lgesql --system lgesql` | `lgesql` |
| 3 Prompted LLMs, 0-shot | `python scripts/run_eval.py --system llm --config configs/baseline_llm.yaml` | `llm-spider-<model>` |
| 3 Prompted LLMs, 3-shot | add `--set shots=3` | `llm-spider-<model>-3shot` |
| 3 Larger model (Voyager) | `--config configs/baseline_voyager.yaml` | `llm-spider-<model>` |
| Ours (base model) | `python scripts/run_eval.py --system pipeline --config configs/pipeline.yaml` | `pipeline-spider-<model>` |
| Ours (QLoRA) | `--config configs/pipeline_qlora.yaml` (docs/QLORA_SERVING.md) | `pipeline-spider-qlora` |

Use `--limit K` for a smoke test; report final numbers on the same K for every system (or full dev).

## Metrics
- **EX / validity / EX by difficulty / latency / model calls**: computed by `run_eval.py`.
- **logprob_coverage**: share of sampled queries that came with token log-probs. Tie-break rule 2
  (confidence) needs them; if this is 0 for a backend, rule 2 never fires there and ties fall
  through to rule 3 (simplicity). Report it next to the `decided_by` counts.
  Difficulty uses `src/text2sql/evaluation/hardness.py`, a re-implementation of the official
  Spider hardness rules (no official parser needed).
- **Exact-set match (EM) and test-suite accuracy**: official scripts.
  ```bash
  python scripts/export_official.py --run runs/<run>
  git clone https://github.com/taoyds/test-suite-sql-eval.git ../test-suite-sql-eval
  cd ../test-suite-sql-eval
  python evaluation.py --gold <repo>/runs/<run>/official/gold.txt --pred <repo>/runs/<run>/official/pred.txt \
      --db <repo>/data/spider/database --table <repo>/data/spider/tables.json --etype all
  ```
  For test-suite accuracy, download the test-suite databases from that repo's README and pass
  their folder as `--db`. Save the printed table to `evaluations/official/<run>.txt`.
- **VES (BIRD)**: computed automatically for `--dataset bird` (mean of 1[correct]·sqrt(t_gold/t_pred), median of 3 timings). The official script repeats more and drops outliers, so cross-check the final number with it.
- **BIRD difficulty**: BIRD's own simple / moderate / challenging labels from `dev.json`.

## Ablations (`scripts/run_ablations.py`)
| Name | Change vs. full pipeline | Question it answers |
|---|---|---|
| full | — | reference |
| no-prune | full schema | does pruning help? |
| single-greedy | N=1, R=0 | the model alone |
| vote-no-repair | R=0 | what does repair add? |
| repair-no-vote | N=1 | what does voting add? |
| full-3shot | 3 shots | do examples help on top? |
| `--sweep` | N in {1,3,5,8}, R in {0,1,2,3} | accuracy vs. latency |

## Error analysis
`python scripts/error_analysis.py --run runs/<run>` labels every wrong prediction
(schema_linking, join, nested_or_aggregation, value_grounding, syntax, other). Review the CSV
and fill `human_label` where the automatic label is wrong before quoting the counts.
