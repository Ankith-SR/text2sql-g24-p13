# Working together

1. `main` is protected by convention: nobody pushes to it directly.
2. Branch per task: `git switch -c <yourname>/<short-task>` (e.g. `divyesh/bird-values`).
3. Small commits, clear messages. Pull often: `git pull --rebase origin main`.
4. Open a PR into `main`, fill the template, get one teammate to review, then squash-merge.
5. Run `make test && make lint` before pushing. CI runs the same.
6. Never commit: datasets, model weights/adapters, `.env`, API keys, big `preds.json` files.
7. Experiments: every run must leave a `run_record.json` + `metrics.json` (done automatically by
   `scripts/run_eval.py` and the training script). Use the same dev sets, prompts and DB files for
   every system, otherwise the comparison is not valid.
8. Notebooks: clear outputs before committing (or keep them out of the repo).

Conflicts: if a rebase conflicts, fix the files, `git add` them, `git rebase --continue`. Ask in the group chat before force-pushing anything.
