# data/

Not tracked by git (see .gitignore). Run `scripts/setup_data.sh` for download steps.
Expected layout: `data/spider/{train_spider.json,dev.json,tables.json,database/}` and
`data/bird/{train,dev}/...`. If your BIRD folders are named differently, edit `load_bird`
in `src/text2sql/data/loaders.py` (one place).
