# Using the QLoRA model in the pipeline (Stage 3, component 2)

1. Train on a GPU node (Sol):
   ```bash
   pip install -r requirements-train.txt
   python -m text2sql.training.qlora_sft --config configs/qlora.yaml --pilot 500   # pilot first
   python -m text2sql.training.qlora_sft --config configs/qlora.yaml               # full run
   ```
   The adapter is saved to `checkpoints/qwen25coder7b-qlora/{pilot,full}/adapter` (git-ignored).
2. Serve base model + adapter with vLLM's OpenAI-compatible server (same GPU node):
   ```bash
   pip install vllm
   vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --enable-lora \
       --lora-modules qlora=checkpoints/qwen25coder7b-qlora/full/adapter --max-lora-rank 16 --port 8000
   ```
3. Point the repo at it (`.env`: `VLLM_BASE_URL=http://<node>:8000/v1`) and run
   ```bash
   python scripts/run_eval.py --system pipeline --config configs/pipeline_qlora.yaml --limit 200
   python scripts/run_eval.py --system pipeline --config configs/pipeline_qlora.yaml --set n_candidates=1 max_repairs=0 --name qlora-alone
   ```
   The second line is "fine-tuned model alone" from the proposal. Compare with the same commands
   using `configs/pipeline.yaml` (untuned base) for the base-vs-QLoRA ablation. Use the same base
   model in both: vLLM serves the Hugging Face Qwen weights, so also run the base model through vLLM
   (`--model Qwen/Qwen2.5-Coder-7B-Instruct --set backend=vllm`) rather than comparing against Ollama's
   quantized build.
