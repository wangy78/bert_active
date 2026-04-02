"""GPU job queue: human_ensembl_regulatory 3-class experiments.

Usage:
    python run_queue_regulatory.py                    # use all 8 GPUs
    python run_queue_regulatory.py --gpus 0 1 2 3     # use specific GPUs
    python run_queue_regulatory.py --dry-run          # print commands only
"""

from run_queue_lib import Job, run_queue

SEEDS = [42, 43, 44]

JOBS = [
    # --- Regulatory q100 r20 (core comparison, all strategies) ---
    Job("configs/regulatory_baseline_random.yaml",    seeds=SEEDS),
    Job("configs/regulatory_baseline_entropy.yaml",   seeds=SEEDS),
    Job("configs/regulatory_baseline_bald.yaml",      seeds=SEEDS),
    Job("configs/regulatory_baseline_badge.yaml",     seeds=SEEDS),
    Job("configs/regulatory_baseline_batch_bald.yaml",seeds=SEEDS),
    Job("configs/regulatory_baseline_coreset.yaml",   seeds=SEEDS),

    # --- Batch size variants: badge ---
    Job("configs/regulatory_baseline_badge.yaml",
        extra_args=["--n-query", "20", "--n-rounds", "100"], seeds=SEEDS),
    Job("configs/regulatory_baseline_badge.yaml",
        extra_args=["--n-query", "50", "--n-rounds",  "40"], seeds=SEEDS),
    Job("configs/regulatory_baseline_badge.yaml",
        extra_args=["--n-query", "200", "--n-rounds", "10"], seeds=SEEDS),

    # --- Batch size variants: batch_bald ---
    Job("configs/regulatory_baseline_batch_bald.yaml",
        extra_args=["--n-query", "20", "--n-rounds", "100"], seeds=SEEDS),
    Job("configs/regulatory_baseline_batch_bald.yaml",
        extra_args=["--n-query", "50", "--n-rounds",  "40"], seeds=SEEDS),
    Job("configs/regulatory_baseline_batch_bald.yaml",
        extra_args=["--n-query", "200", "--n-rounds", "10"], seeds=SEEDS),
]

if __name__ == "__main__":
    run_queue(JOBS, default_log_dir="logs/regulatory")
