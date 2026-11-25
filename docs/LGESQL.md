# LGESQL reproduction (Tier 2, owner: Hui Yu)
Public code: https://github.com/rhythmcao/text2sql-lgesql
Clone it **outside** this repo, follow its README to train/evaluate on Spider, then score its dev
predictions (one SQL per line, dev order) with the same metrics as every other tier:
```bash
python scripts/score_preds.py --preds ../text2sql-lgesql/<pred_file>.txt --name lgesql --system lgesql
make summary
``` Fallbacks if too expensive: BiLSTM seq2seq or the
open-source Flan-T5 text2sql baseline (per the proposal).
