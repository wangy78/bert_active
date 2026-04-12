"""Job queue for mm->hs transfer + D-Optimal experiments.

Conditions:
  1. hs_baseline_random        — Human from scratch, random (no transfer)
  2. mm2hs_transfer_random     — Mouse pretrain -> Human, full fine-tune, random
  3. mm2hs_transfer_doptimal   — Mouse pretrain -> Human, full fine-tune, D-Optimal (control)
  4. mm2hs_frozen_doptimal     — Mouse pretrain -> Human, frozen backbone, D-Optimal (key test)

Usage:
  python run_queue_mm2hs.py --gpus 0 1 2 3
  python run_queue_mm2hs.py --gpus 0 --dry-run
"""

from run_queue_lib import Job, run_queue

SEEDS = [42, 43, 44]

jobs = [
    Job("configs/promoter_hs_baseline_random.yaml",      seeds=SEEDS),
    Job("configs/promoter_mm2hs_transfer_random.yaml",   seeds=SEEDS),
    Job("configs/promoter_mm2hs_transfer_doptimal.yaml", seeds=SEEDS),
    Job("configs/promoter_mm2hs_frozen_doptimal.yaml",   seeds=SEEDS),
]

if __name__ == "__main__":
    run_queue(jobs, default_log_dir="logs/mm2hs")
