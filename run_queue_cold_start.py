"""GPU job queue: cold-start comparison (Random vs CoreSet vs TypiClust).

Scientific question: At small budgets (n_init=100, n_query=100), does
TypiClust (typical / high-density selection) outperform CoreSet (diversity)
and Random on DNA classification tasks?

Existing baselines reused from previous runs (not re-run here):
  - baseline_mouse_random   × seeds 42,43,44  (promoter/mm)
  - baseline_mouse_coreset  × seeds 42,43,44  (promoter/mm)
  - nt_enhancer_baseline_random × seeds 42,43,44

New jobs (9 total):
  - baseline_mouse_typiclust     × 3 seeds  (promoter/mm)
  - nt_enhancer_baseline_coreset × 3 seeds  (nt_enhancers)
  - nt_enhancer_baseline_typiclust × 3 seeds (nt_enhancers)

Usage:
    python run_queue_cold_start.py                # all available GPUs
    python run_queue_cold_start.py --gpus 0 1     # specific GPUs
    python run_queue_cold_start.py --dry-run      # print commands only
"""

from run_queue_lib import Job, run_queue

SEEDS = [42, 43, 44]

JOBS = [
    # Mouse promoter: only typiclust is new (random+coreset already done)
    Job("configs/promoter_baseline_typiclust.yaml", seeds=SEEDS),

    # NT enhancers: coreset and typiclust are new (random already done)
    Job("configs/nt_enhancer_baseline_coreset.yaml",   seeds=SEEDS),
    Job("configs/nt_enhancer_baseline_typiclust.yaml", seeds=SEEDS),
]

if __name__ == "__main__":
    run_queue(JOBS, default_log_dir="logs/cold_start")
