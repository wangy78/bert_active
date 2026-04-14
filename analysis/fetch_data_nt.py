"""
Fetch NT cross-task transfer experiment runs from W&B and cache locally.

Discovers runs by tag "nt_downstream". Saves separate CSVs for:
  - analysis/data/nt_enhancer_runs.csv    (A1, binary)
  - analysis/data/nt_enhancer_types_runs.csv  (A2, 3-class)
"""

import re
import wandb
import pandas as pd
from pathlib import Path

api = wandb.Api()
ENTITY  = "codycjy"
PROJECT = "bert-active-dna"
CACHE_DIR = Path("analysis/data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["al/round", "al/eval_accuracy", "al/eval_f1_macro", "al/n_labeled"]

# Condition name derived from experiment_name, e.g.:
#   nt_enhancer_baseline_random  -> baseline_random
#   nt_enhancer_frozen_doptimal  -> frozen_doptimal
def parse_condition(exp_name: str, prefix: str) -> str:
    cond = exp_name
    cond = re.sub(rf"^{re.escape(prefix)}_", "", cond)
    return cond

runs = api.runs(
    f"{ENTITY}/{PROJECT}",
    filters={"tags": {"$in": ["nt_downstream"]}, "state": "finished"},
)
print(f"Found {len(runs)} finished nt_downstream runs")

rows_a1, rows_a2 = [], []

for run in runs:
    # Determine target from tags
    if "h3k4me3_to_enhancer_types" in run.tags:
        target = "nt_enhancer_types"
        prefix = "nt_enhancer_types"
        rows = rows_a2
    elif "h3k4me3_to_enhancer" in run.tags:
        target = "nt_enhancers"
        prefix = "nt_enhancer"
        rows = rows_a1
    else:
        print(f"  skip {run.name} (no target tag)")
        continue

    # Parse seed from run name (e.g. nt_enhancer_transfer_badge_seed42)
    seed_m = re.search(r"seed(\d+)$", run.name)
    if not seed_m:
        print(f"  skip {run.name} (no seed)")
        continue
    seed = int(seed_m.group(1))

    # Get condition from run config experiment_name
    exp_name = run.config.get("experiment_name", run.name.rsplit("_seed", 1)[0])
    condition = parse_condition(exp_name, prefix)

    cache_file = CACHE_DIR / f"nt_{run.id}.csv"
    if cache_file.exists():
        hist = pd.read_csv(cache_file)
        print(f"  {run.name} cached")
    else:
        hist = run.history(keys=KEYS, pandas=True)
        hist = hist.dropna(subset=["al/round"]).sort_values("al/round").reset_index(drop=True)
        hist.to_csv(cache_file, index=False)
        print(f"  {run.name} fetched ({len(hist)} rows)")

    hist["condition"] = condition
    hist["target"]    = target
    hist["seed"]      = seed
    hist["n_labeled"] = hist["al/n_labeled"].astype(int)
    rows.append(hist)

# Save per-target CSVs
if rows_a1:
    df_a1 = pd.concat(rows_a1, ignore_index=True)
    df_a1.to_csv(CACHE_DIR / "nt_enhancer_runs.csv", index=False)
    print(f"\nA1 saved {len(df_a1)} rows → analysis/data/nt_enhancer_runs.csv")
    print("Conditions:", sorted(df_a1["condition"].unique()))
else:
    print("\nNo A1 (enhancers) runs found")

if rows_a2:
    df_a2 = pd.concat(rows_a2, ignore_index=True)
    df_a2.to_csv(CACHE_DIR / "nt_enhancer_types_runs.csv", index=False)
    print(f"\nA2 saved {len(df_a2)} rows → analysis/data/nt_enhancer_types_runs.csv")
    print("Conditions:", sorted(df_a2["condition"].unique()))
else:
    print("\nNo A2 (enhancer_types) runs found")
