"""Job queue for mm->hs small-batch (q=20) experiments.

Conditions (vs q=100 baseline already run):
  - mm2hs_transfer_random_q20   : full FT + random,     q=20 r=100
  - mm2hs_frozen_doptimal_q20   : frozen + D-Optimal,   q=20 r=100

Usage:
  python run_queue_mm2hs_q20.py --gpus 0 1 2 3
  python run_queue_mm2hs_q20.py --dry-run
"""

from run_queue_lib import Job, run_queue

SEEDS = [42, 43, 44]

jobs = [
    Job("configs/promoter_mm2hs_transfer_random_q20.yaml",   seeds=SEEDS),
    Job("configs/promoter_mm2hs_frozen_doptimal_q20.yaml",   seeds=SEEDS),
]

if __name__ == "__main__":
    run_queue(jobs, default_log_dir="logs/mm2hs_q20")
