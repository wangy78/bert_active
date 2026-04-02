"""GPU job queue: all D-Optimal / DOE experiments (both datasets).

2x2 design per dataset:
  random init  + random query   → baseline (from main queues, not repeated here)
  random init  + doptimal query → *_baseline_doptimal        (isolates query effect)
  doptimal init + random query  → *_doptimal_init_random     (isolates init effect)
  doptimal init + doptimal query→ *_baseline_doptimal        (full DOE)
  doptimal init + badge query   → *_doptimal_init_badge      (best of both)

Datasets: mouse promoter (binary) + fungi species (20-class)

Usage:
    python run_queue_doe.py                    # use all 8 GPUs
    python run_queue_doe.py --gpus 0 1 2 3     # use specific GPUs
    python run_queue_doe.py --dry-run          # print commands only
"""

from run_queue_lib import Job, run_queue

SEEDS = [42, 43, 44]

N_INITS = [100, 300, 500]

JOBS = [
    # ── Mouse promoter (binary) ───────────────────────────────────────────────
    Job("configs/promoter_baseline_doptimal.yaml",    seeds=SEEDS),
    Job("configs/promoter_doptimal_init_random.yaml", seeds=SEEDS),
    Job("configs/promoter_doptimal_init_badge.yaml",  seeds=SEEDS),

    # n_init sweep: doptimal init + badge query
    *[
        Job("configs/promoter_doptimal_init_badge.yaml",
            extra_args=["--n-init", str(n)], seeds=SEEDS)
        for n in N_INITS if n != 100
    ],

    # ── Fungi species 20-class ────────────────────────────────────────────────
    Job("configs/fungi_baseline_doptimal.yaml",       seeds=SEEDS),
    Job("configs/fungi_doptimal_init_random.yaml",    seeds=SEEDS),
    Job("configs/fungi_doptimal_init_badge.yaml",     seeds=SEEDS),

    # n_init sweep: doptimal init + badge query
    *[
        Job("configs/fungi_doptimal_init_badge.yaml",
            extra_args=["--n-init", str(n)], seeds=SEEDS)
        for n in N_INITS if n != 100
    ],
]

if __name__ == "__main__":
    run_queue(JOBS, default_log_dir="logs/doe")
