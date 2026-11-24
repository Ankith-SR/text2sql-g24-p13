# Task board (mirrors the proposal; create GitHub issues with `bash scripts/create_issues.sh`)

Fixed milestones: presentation Nov 12 (done), demo Dec 3, final report Dec 8.
Plan dates from the proposal: baselines Oct 9-20, error analysis + pruning Oct 21-Nov 5,
tune/vote/repair Nov 6-20, ablations + evaluation Nov 21-28. Update the status below honestly.

## Ankith
- [ ] Prompted baselines on Spider dev: Qwen2.5-Coder-7B and a Llama-family model, 0-shot and 3-shot
- [ ] One larger open model via ASU Voyager as upper reference
- [ ] QLoRA pilot (500 rows) vs untuned base, then full Spider run
- [ ] Phase 2: add BIRD train split to fine-tuning
## Divyesh
- [ ] Download Spider + BIRD, verify `loaders.py` folder names, share Sol paths with team
- [ ] Evaluate BM25 pruning (table/column recall on dev); replace with a better ranker if recall is low
- [ ] BIRD value retrieval (BM25 over cell values), check speed on large tables
## Harshan
- [ ] Run vote + tie-break on real candidates; log which tie rule decides how often
- [ ] Repair loop on real failures; sweep N and R for accuracy vs latency
- [ ] Demo app: DB picker, candidate/vote inspector, latency readout
## Hui Yu
- [ ] Heuristic baseline on Spider dev (already coded; run and record)
- [ ] LGESQL reproduction (see docs/LGESQL.md)
- [ ] Official EM + test-suite accuracy wrappers; VES for BIRD
- [ ] Error analysis: label every wrong dev prediction (analysis/error_taxonomy.py + human review)
## All
- [ ] Ablations: full vs pruned (+/- values), base vs QLoRA, single vs N-vote, vote +/- repair
- [ ] Demo (Dec 3) and final report (Dec 8)
