"""Shared utilities for GPU job queues.

Import this in any run_queue_*.py:

    from run_queue_lib import Job, run_queue
"""

import argparse
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Job:
    config: str
    extra_args: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42])

    def commands(self) -> list[tuple[str, list[str]]]:
        """Returns list of (label, argv) for each seed."""
        cmds = []
        for seed in self.seeds:
            name = Path(self.config).stem
            args = ["--config", self.config, "--seed", str(seed)] + self.extra_args
            label = f"{name}_seed{seed}"
            if self.extra_args:
                pairs = zip(self.extra_args[::2], self.extra_args[1::2])
                suffix = "_".join(v.lstrip("-").replace("-", "") for _, v in pairs)
                label = f"{name}_{suffix}_seed{seed}"
            cmds.append((label, args))
        return cmds


def build_queue(jobs: list[Job]) -> list[tuple[str, list[str]]]:
    queue = []
    for job in jobs:
        queue.extend(job.commands())
    return queue


def _launch(gpu: int, label: str, argv: list[str], log_dir: Path) -> subprocess.Popen:
    log_file = log_dir / f"{label}.log"
    cmd = ["python", "-m", "bert_active.run"] + argv
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"  [GPU {gpu}] START  {label}  → {log_file.name}")
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
    )


def run_queue(jobs: list[Job], default_log_dir: str = "logs") -> None:
    """Parse args and run a job queue across GPUs.

    Args:
        jobs: List of Job objects to run.
        default_log_dir: Default log directory (overridable via --log-dir).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--log-dir", type=str, default=default_log_dir)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    queue = build_queue(jobs)
    total = len(queue)

    print(f"{'='*60}")
    print(f"Total jobs : {total}")
    print(f"GPUs       : {args.gpus}")
    print(f"Log dir    : {log_dir}")
    print(f"{'='*60}")

    if args.dry_run:
        for label, argv in queue:
            print(f"  {label}  →  python -m bert_active.run {' '.join(argv)}")
        return

    running: dict[int, tuple[subprocess.Popen, str, float]] = {}
    queue_iter = iter(queue)
    done = 0

    def try_fill_gpus():
        for gpu in args.gpus:
            if gpu not in running:
                try:
                    label, argv = next(queue_iter)
                    running[gpu] = (_launch(gpu, label, argv, log_dir), label, time.time())
                except StopIteration:
                    pass

    try_fill_gpus()

    while running:
        time.sleep(10)
        finished_gpus = []
        for gpu, (proc, label, t0) in running.items():
            ret = proc.poll()
            if ret is not None:
                elapsed = (time.time() - t0) / 60
                status = "✓" if ret == 0 else f"✗(exit {ret})"
                print(f"  [GPU {gpu}] DONE   {label}  {status}  {elapsed:.1f}min  ({done+1}/{total})")
                done += 1
                finished_gpus.append(gpu)
        for gpu in finished_gpus:
            del running[gpu]
        try_fill_gpus()

    print(f"\nAll {total} jobs finished.  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
