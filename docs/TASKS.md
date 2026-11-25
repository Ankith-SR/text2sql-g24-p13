# Task board (mirrors the proposal; create GitHub issues with `bash scripts/create_issues.sh`)

Fixed milestones: presentation Nov 12 (done), **demo Nov 30** (course schedule, Group 24), final report Dec 8.
Plan dates from the proposal: baselines Oct 9-20, error analysis + pruning Oct 21-Nov 5,
tune/vote/repair Nov 6-20, ablations + evaluation Nov 21-28. Update the status below honestly.

## Ankith
- [x] Prompted-baseline code: multi-model, 0/3-shot (`configs/baseline_llm.yaml`, `--set shots=3`)
- [ ] Run prompted baselines on Spider dev: Qwen2.5-Coder-7B and Llama-3.1-8B, 0-shot and 3-shot
- [ ] One larger open model via ASU Voyager (`configs/baseline_voyager.yaml`, fill model id)
- [ ] QLoRA pilot (500 rows) vs untuned base, then full Spider run (`docs/QLORA_SERVING.md`)
- [ ] Phase 2: add BIRD train split to fine-tuning
## Divyesh
- [ ] Download Spider + BIRD, verify `loaders.py` folder names, share Sol paths with team
- [ ] `python scripts/make_samples.py` and commit `data/samples/spider`
- [ ] Evaluate BM25 pruning (table/column recall on dev); replace with a better ranker if recall is low
- [ ] BIRD value retrieval (BM25 over cell values), check speed on large tables
## Harshan
- [x] Sampling that really returns N candidates (loop when the backend ignores `n`)
- [x] Demo app: DB picker, example questions, candidate/vote inspector, latency readout
- [ ] Run vote + tie-break on real candidates; report `decided_by` counts from metrics.json
- [ ] Repair loop on real failures; sweep N and R for accuracy vs latency (`run_ablations.py --sweep`)
## Hui Yu
- [x] Spider difficulty levels (`evaluation/hardness.py`), error-analysis script, official EM/test-suite export
- [ ] Heuristic baseline on Spider dev (`make heuristic`)
- [ ] LGESQL reproduction (see docs/LGESQL.md)
- [ ] EM + test-suite numbers via `scripts/export_official.py`; VES for BIRD
- [ ] Error analysis on every wrong dev prediction (+ human review of `errors.csv`)
## All
- [x] Ablation runner (`scripts/run_ablations.py`) and results summary (`make summary`)
- [ ] Run ablations: full vs pruned (+/- values), base vs QLoRA, single vs N-vote, vote +/- repair
- [ ] Demo slides (Nov 30, PDF `24-13-FA26-Group-DEMO-Presentation.pdf`) and final report (Dec 8)
