"""List model ids an OpenAI-compatible backend serves (to fill configs/*.yaml).

  python scripts/list_models.py --backend voyager
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.baselines.llm_prompt import make_client  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="ollama", choices=["ollama", "voyager", "vllm"])
args = ap.parse_args()
for m in make_client(args.backend).models.list():
    print(m.id)
