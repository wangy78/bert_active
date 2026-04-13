"""Zero-shot cross-species evaluation: mouse-trained model on human promoter data.

Answers the question: does the mouse-pretrained model already generalise to human
promoters *without* any human-labeled data (n_labeled=0)?

Pipeline:
  1. Load DNABERT-2
  2. Fine-tune on full mouse dataset (same pretrain config as transfer runs)
  3. Evaluate directly on human test set  ← zero-shot transfer score
  4. Log result to WandB

Compare this number against:
  - Human from-scratch baseline at n=100 (first AL round)
  - mm->hs transfer curves at n=100

Usage:
  python eval_zeroshot_mm2hs.py
  python eval_zeroshot_mm2hs.py --seeds 42 43 44 --no-wandb
"""

import argparse
import logging
from pathlib import Path

import wandb
from bert_active.config.experiment import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    TrainerConfig,
    PretrainConfig,
    WandbConfig,
)
from bert_active.data.promoter_dataset import load_transfer_data
from bert_active.data.tokenization import build_dataset, create_tokenizer
from bert_active.engine.trainer import Trainer
from bert_active.models.classifier import ModelWrapper, create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_zeroshot(seed: int, use_wandb: bool) -> dict[str, float]:
    log.info(f"=== Zero-shot mm->hs  seed={seed} ===")

    model_cfg   = ModelConfig(
        name="zhihan1996/DNABERT-2-117M",
        num_labels=2,
        attn_implementation="flash_attention_2",
        torch_dtype="float32",
        max_length=128,
    )
    pretrain_cfg = PretrainConfig(
        enabled=True,
        source_species="mm",
        num_epochs=10,
        learning_rate=2e-5,
        batch_size=16,
    )
    trainer_cfg = TrainerConfig(
        learning_rate=2e-5,
        weight_decay=0.01,
        batch_size=16,
        eval_batch_size=32,
        num_epochs=10,       # same as pretrain epochs
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        device="cuda",
    )

    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project="bert-active-dna",
            entity="codycjy",
            name=f"mm2hs_zeroshot_seed{seed}",
            config={
                "experiment": "zero_shot_mm2hs",
                "seed": seed,
                "pretrain_epochs": pretrain_cfg.num_epochs,
                "n_labeled_human": 0,
            },
            tags=["promoter", "transfer", "mm2hs", "zero-shot", f"seed{seed}"],
            notes="Mouse-trained model evaluated directly on human test set, no human labels",
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading mouse (source) and human (target) data...")
    source_texts, source_labels, _, target_test_ds, target_test_texts = load_transfer_data(
        source_species="mm",
        target_species="hs",
        data_dir=".",
        seed=seed,
    )

    tokenizer = create_tokenizer(model_cfg.name)

    source_dataset = build_dataset(
        tokenizer=tokenizer,
        texts=source_texts,
        labels=source_labels,
        max_length=model_cfg.max_length,
    )
    human_test_dataset = build_dataset(
        tokenizer=tokenizer,
        texts=target_test_texts,
        labels=target_test_ds.labels,
        max_length=model_cfg.max_length,
    )

    # ── Train on mouse (or load cached checkpoint) ────────────────────────────
    ckpt_dir = Path("checkpoints") / f"pretrained_mm_seed{seed}"
    model = create_model(
        model_name=model_cfg.name,
        num_labels=model_cfg.num_labels,
        attn_implementation=model_cfg.attn_implementation,
        torch_dtype=model_cfg.torch_dtype,
    )
    model_wrapper = ModelWrapper(model=model, device=trainer_cfg.device)

    ckpt_file = ckpt_dir / "model_state_dict.pt"
    if ckpt_file.exists():
        log.info(f"Loading pretrained mouse weights from {ckpt_file}")
        import torch
        state_dict = torch.load(str(ckpt_file), map_location=model_wrapper.device)
        model_wrapper.model.load_state_dict(state_dict)
    else:
        log.info(f"Pretraining on mouse: {len(source_texts)} sequences, {pretrain_cfg.num_epochs} epochs")
        trainer = Trainer(model=model_wrapper, config=trainer_cfg, wandb_run=wandb_run)
        trainer.train(source_dataset)
        import torch
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_wrapper.model.state_dict(), str(ckpt_file))
        log.info(f"Saved pretrained weights → {ckpt_file}")

    # ── Evaluate on human (zero-shot) ─────────────────────────────────────────
    log.info("Evaluating on human test set (zero-shot)...")
    metrics = trainer.evaluate(human_test_dataset)

    log.info(
        f"Zero-shot mm->hs  seed={seed}:  "
        f"acc={metrics['eval_accuracy']:.4f}  "
        f"f1={metrics['eval_f1_macro']:.4f}"
    )

    if wandb_run is not None:
        wandb_run.summary.update({
            "zeroshot/eval_accuracy": metrics["eval_accuracy"],
            "zeroshot/eval_f1_macro": metrics["eval_f1_macro"],
            "zeroshot/n_labeled_human": 0,
        })
        wandb_run.finish()

    return {"seed": seed, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        r = run_zeroshot(seed=seed, use_wandb=not args.no_wandb)
        results.append(r)

    print(f"\n{'='*55}")
    print(f"{'Seed':<8} {'Accuracy':>10} {'F1 Macro':>10}")
    print(f"{'='*55}")
    for r in results:
        print(f"{r['seed']:<8} {r['eval_accuracy']:>10.4f} {r['eval_f1_macro']:>10.4f}")

    if len(results) > 1:
        import numpy as np
        f1s = [r["eval_f1_macro"] for r in results]
        print(f"{'mean':<8} {sum(r['eval_accuracy'] for r in results)/len(results):>10.4f} {float(np.mean(f1s)):>10.4f}")
        print(f"{'std':<8} {'':>10} {float(np.std(f1s)):>10.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
