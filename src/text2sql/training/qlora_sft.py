"""Stage 3, component 2: QLoRA fine-tune Qwen2.5-Coder-7B on pruned prompts (owner: Ankith).

Run (GPU box / Sol cluster, after `pip install -r requirements-train.txt`):
  python -m text2sql.training.qlora_sft --config configs/qlora.yaml --pilot 500

Follows the team guidelines doc: read the model card, use its tokenizer/chat template,
start with a pilot subset, compare against the untuned base on the same prompts, and
keep a run record (model/data revisions, config, seed, hardware, eval results).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from text2sql.baselines.llm_prompt import SYSTEM, build_prompt
from text2sql.data.loaders import load_bird, load_spider
from text2sql.data.schema import load_schemas
from text2sql.pruning.schema_pruning import prune
from text2sql.utils.run_record import write_run_record

# Training uses exactly the system prompt and user prompt (build_prompt) used at inference, so
# base vs. QLoRA is compared "on the same prompts" (proposal, Stage 3).


def build_sft_rows(cfg: dict, pilot: int | None, seed: int) -> list[dict]:
    rows = []
    spider_root = Path(cfg["data"]["spider_root"])
    schemas = load_schemas(spider_root / "tables.json")
    for ex in load_spider(spider_root, "train"):
        sch = schemas[ex.db_id]
        keep = prune(ex.question, sch) if cfg["data"].get("prune", True) else None
        user = build_prompt(sch.to_prompt(keep), ex.question)
        rows.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": ex.gold_sql},
        ]})
    if cfg["data"].get("bird_root"):  # phase 2: add BIRD once Spider results are in
        bird_root = Path(cfg["data"]["bird_root"])
        bschemas = load_schemas(bird_root / "train" / "train_tables.json")
        for ex in load_bird(bird_root, "train"):
            sch = bschemas[ex.db_id]
            keep = prune(ex.question, sch) if cfg["data"].get("prune", True) else None
            user = build_prompt(sch.to_prompt(keep), ex.question, ex.evidence)
            rows.append({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": ex.gold_sql},
            ]})
    random.Random(seed).shuffle(rows)
    return rows[:pilot] if pilot else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qlora.yaml")
    ap.add_argument("--pilot", type=int, default=None, help="train on a small subset first")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 42)

    # Heavy imports here so the rest of the repo works without GPU deps.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    out_dir = Path(cfg["output_dir"]) / ("pilot" if args.pilot else "full")
    rows = build_sft_rows(cfg, args.pilot, seed)
    ds = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"], task_type="CAUSAL_LM",
                      target_modules="all-linear")
    train_args = SFTConfig(
        output_dir=str(out_dir), seed=seed,
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"], num_train_epochs=cfg["train"]["epochs"],
        logging_steps=10, save_strategy="epoch", bf16=True,
        model_init_kwargs={"quantization_config": bnb, "torch_dtype": torch.bfloat16},
    )
    trainer = SFTTrainer(model=cfg["base_model"], args=train_args, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    write_run_record(out_dir, cfg, extra={"pilot": args.pilot, "n_rows": len(rows)})


if __name__ == "__main__":
    main()
