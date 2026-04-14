"""Job queue for NT cross-task transfer experiments.

Conditions (per target):
  1. baseline_random        — From scratch, random AL (no transfer)
  2. transfer_random        — H3K4me3 pretrain, full fine-tune, random AL
  3. transfer_badge         — H3K4me3 pretrain, full fine-tune, BADGE AL
  4. frozen_badge           — H3K4me3 pretrain, frozen backbone, BADGE AL
  5. frozen_doptimal        — H3K4me3 pretrain, frozen backbone, D-Optimal AL

Two targets:
  A1: nt_enhancers (binary)
  A2: nt_enhancer_types (3-class)

Total: 10 configs × 3 seeds = 30 jobs

Usage:
  python run_queue_nt_transfer.py --gpus 0 1 2 3
  python run_queue_nt_transfer.py --gpus 0 --dry-run
"""

from run_queue_lib import Job, run_queue

SEEDS = [42, 43, 44]

jobs = [
    # A1: enhancers (binary)
    Job("configs/nt_enhancer_baseline_random.yaml",     seeds=SEEDS),
    Job("configs/nt_enhancer_transfer_random.yaml",     seeds=SEEDS),
    Job("configs/nt_enhancer_transfer_badge.yaml",      seeds=SEEDS),
    Job("configs/nt_enhancer_frozen_badge.yaml",        seeds=SEEDS),
    Job("configs/nt_enhancer_frozen_doptimal.yaml",     seeds=SEEDS),
    # A2: enhancers_types (3-class)
    Job("configs/nt_enhancer_types_baseline_random.yaml",     seeds=SEEDS),
    Job("configs/nt_enhancer_types_transfer_random.yaml",     seeds=SEEDS),
    Job("configs/nt_enhancer_types_transfer_badge.yaml",      seeds=SEEDS),
    Job("configs/nt_enhancer_types_frozen_badge.yaml",        seeds=SEEDS),
    Job("configs/nt_enhancer_types_frozen_doptimal.yaml",     seeds=SEEDS),
]

if __name__ == "__main__":
    run_queue(jobs, default_log_dir="logs/nt_transfer")
