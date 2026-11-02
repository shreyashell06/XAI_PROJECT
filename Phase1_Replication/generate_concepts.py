"""(Optional) Regenerate a concept set with an LLM (Label-free CBM step 1) and filter it (FVLC App. C).

The main experiments use the GPT-3 concept sets released with Label-free CBM (data/concept_sets/),
because GPT-3 (text-davinci) is no longer available. This script reproduces the procedure with any
OpenAI-compatible chat model, e.g.:

  export OPENAI_API_KEY=...
  python generate_concepts.py --dataset cifar10 --model gpt-4o-mini --out data/concept_sets/cifar10_new.txt
  # Groq / Together / local vLLM: add --base_url https://api.groq.com/openai/v1 --model llama-3.1-8b-instant

  python run_experiments.py --dataset cifar10 --concept_set data/concept_sets/cifar10_new.txt ...
"""
import argparse
import json

from fvlc.concepts import filter_concepts, generate_concepts
from fvlc.data import get_classes
from fvlc.utils import write_lines

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--model", default="gpt-4o-mini")
ap.add_argument("--base_url", default=None)
ap.add_argument("--prompts", nargs="+", default=None,
                help="subset of: important around superclass (paper: CUB uses only 'important')")
ap.add_argument("--out", required=True)
a = ap.parse_args()

classes = get_classes(a.dataset)
prompts = a.prompts or (["important"] if a.dataset == "cub" else ["important", "around", "superclass"])
raw = generate_concepts(classes, model=a.model, prompts=prompts, base_url=a.base_url)
with open(a.out.replace(".txt", "_raw.json"), "w") as f:
    json.dump(raw, f, indent=2)
concepts = filter_concepts([c for v in raw.values() for c in v], classes)
write_lines(a.out, concepts)
print(f"{sum(len(v) for v in raw.values())} raw -> {len(concepts)} filtered concepts -> {a.out}")
