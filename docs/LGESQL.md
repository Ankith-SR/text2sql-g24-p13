# LGESQL reproduction (Tier 2, owner: Hui Yu)
Public code: https://github.com/rhythmcao/text2sql-lgesql
Clone it **outside** this repo, follow its README to train/evaluate on Spider, then convert its
predictions to `{qid: sql}` JSON and score them with `text2sql.evaluation.execution.evaluate` so the
numbers are comparable with the other tiers. Fallbacks if too expensive: BiLSTM seq2seq or the
open-source Flan-T5 text2sql baseline (per the proposal).
