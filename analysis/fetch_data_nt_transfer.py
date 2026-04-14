"""
Fetch all NT cross-task transfer run histories from WandB and cache locally.
Discovers runs by the 'nt_downstream' tag.
"""

import wandb
import pandas as pd
from pathlib import Path

api = wandb.Api()
ENTITY  = "codycjy"
PROJECT = "bert-active-dna"
CACHE_DIR = Path("analysis/data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["al/round", "al/eval_accuracy", "al/eval_f1_macro", "al/n_labeled"]

# Experiment name -> (target, condition) mapping
CONDITION_MAP = {
    "nt_enhancer_baseline_random":        ("enhancers",       "baseline_random"),
    "nt_enhancer_transfer_random":        ("enhancers",       "transfer_random"),
    "nt_enhancer_transfer_badge":         ("enhancers",       "transfer_badge"),
    "nt_enhancer_frozen_badge":           ("enhancers",       "frozen_badge"),
    "nt_enhancer_frozen_doptimal":        ("enhancers",       "frozen_doptimal"),
    "nt_enhancer_types_baseline_random":  ("enhancer_types",  "baseline_random"),
    "nt_enhancer_types_transfer_random":  ("enhancer_types",  "transfer_random"),
    "nt_enhancer_types_transfer_badge":   ("enhancer_types",  "transfer_badge"),
    "nt_enhancer_types_frozen_badge":     ("enhancer_types",  "frozen_badge"),
    "nt_enhancer_types_frozen_doptimal":  ("enhancer_types",  "frozen_doptimal"),
}

runs = api.runs(
    f"{ENTITY}/{PROJECT}",
    filters={"tags": {"$in": ["nt_downstream"]}, "state": "finished"},
)

print(f"Found {len(runs)} finished nt_downstream runs")

rows = []
for run in runs:
    cfg = run.config
    exp_name = cfg.get("experiment_name", "")
    seed     = cfg.get("data", {}).get("seed", None)

    if exp_name not in CONDITION_MAP:
        print(f"  skip {run.name} (unknown experiment_name: {exp_name!r})")
        continue
    if seed is None:
        print(f"  skip {run.name} (no seed in config)")
        continue

    target, condition = CONDITION_MAP[exp_name]

    cache_file = CACHE_DIR / f"nt_transfer_{run.id}.csv"
    if cache_file.exists():
        hist = pd.read_csv(cache_file)
        print(f"  {run.name} cached")
    else:
        hist = run.history(keys=KEYS, pandas=True)
        hist = hist.dropna(subset=["al/round"]).sort_values("al/round").reset_index(drop=True)
        hist.to_csv(cache_file, index=False)
        print(f"  {run.name} fetched ({len(hist)} rows)")

    hist["target"]    = target
    hist["condition"] = condition
    hist["seed"]      = int(seed)
    hist["n_labeled"] = hist["al/n_labeled"].astype(int)
    rows.append(hist)

if not rows:
    print("No rows collected — check that runs have finished and tags are correct.")
    raise SystemExit(1)

df = pd.concat(rows, ignore_index=True)
df.to_csv(CACHE_DIR / "nt_transfer_runs.csv", index=False)
print(f"\nSaved {len(df)} rows → analysis/data/nt_transfer_runs.csv")
print("\nTargets found:",    sorted(df["target"].unique()))
print("Conditions found:", sorted(df["condition"].unique()))
print("Seeds found:",      sorted(df["seed"].unique()))
