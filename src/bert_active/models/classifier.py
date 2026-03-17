"""DistilBERT classifier wrapper with embedding and gradient extraction for AL strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from bert_active.data.dataset import TextClassificationDataset


def create_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 4,
    attn_implementation: str = "flash_attention_2",
    torch_dtype: str = "float16",
) -> PreTrainedModel:
    """Create a DistilBERT model for sequence classification with Flash Attention 2."""
    dtype_map: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        # attn_implementation=attn_implementation,
        torch_dtype=dtype_map.get(torch_dtype, torch.float16),
    )
    return model


class ModelWrapper:
    """Wraps a DistilBERT classifier to provide inference utilities needed by AL strategies.

    Provides:
    - predict_proba: softmax probabilities
    - predict_proba_dropout: MC Dropout probabilities
    - get_embeddings: penultimate layer (pre_classifier output)
    - get_grad_embeddings: gradient embeddings for BADGE
    """

    def __init__(self, model: PreTrainedModel, device: str = "cuda") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model = self.model.to(self.device)  # type: ignore[assignment]

    @property
    def num_labels(self) -> int:
        return self.model.config.num_labels  # type: ignore[no-any-return]

    @property
    def embedding_dim(self) -> int:
        return self.model.config.dim  # type: ignore[no-any-return]

    def _make_loader(
        self,
        dataset: TextClassificationDataset,
        batch_size: int = 64,
    ) -> DataLoader[dict[str, Any]]:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    @torch.no_grad()
    def predict_proba(
        self,
        dataset: TextClassificationDataset,
        batch_size: int = 64,
    ) -> NDArray[np.floating[Any]]:
        """Get softmax probabilities for all samples."""
        self.model.eval()
        all_probs: list[NDArray[np.floating[Any]]] = []
        loader = self._make_loader(dataset, batch_size)

        for batch in tqdm(loader, desc="Predicting", leave=False):
            input_ids = batch["input_ids"].to(self.device)  # type: ignore[union-attr]
            attention_mask = batch["attention_mask"].to(self.device)  # type: ignore[union-attr]
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1)
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    @torch.no_grad()
    def predict_proba_dropout(
        self,
        dataset: TextClassificationDataset,
        n_drop: int = 10,
        batch_size: int = 64,
    ) -> NDArray[np.floating[Any]]:
        """MC Dropout: run n_drop forward passes with dropout enabled, return mean probs.

        Returns shape (n_samples, n_classes).
        Also stores individual predictions in self._mc_probs for BALD computation.
        """
        self.model.eval()
        _enable_dropout(self.model)

        n_samples = len(dataset)
        n_classes = self.num_labels
        all_probs = np.zeros((n_drop, n_samples, n_classes), dtype=np.float64)
        loader = self._make_loader(dataset, batch_size)

        for drop_i in range(n_drop):
            offset = 0
            for batch in tqdm(loader, desc=f"MC Dropout {drop_i + 1}/{n_drop}", leave=False):
                input_ids = batch["input_ids"].to(self.device)  # type: ignore[union-attr]
                attention_mask = batch["attention_mask"].to(self.device)  # type: ignore[union-attr]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()
                bsz = probs.shape[0]
                all_probs[drop_i, offset : offset + bsz] = probs
                offset += bsz

        # Store for BALD computation
        self._mc_probs: NDArray[np.floating[Any]] = all_probs
        return all_probs.mean(axis=0)

    @torch.no_grad()
    def get_embeddings(
        self,
        dataset: TextClassificationDataset,
        batch_size: int = 64,
    ) -> NDArray[np.floating[Any]]:
        """Extract penultimate layer embeddings (pre_classifier output).

        For DistilBERT: hidden[:, 0] -> pre_classifier -> ReLU -> this output.
        """
        self.model.eval()
        all_embeddings: list[NDArray[np.floating[Any]]] = []
        loader = self._make_loader(dataset, batch_size)

        for batch in tqdm(loader, desc="Extracting embeddings", leave=False):
            input_ids = batch["input_ids"].to(self.device)  # type: ignore[union-attr]
            attention_mask = batch["attention_mask"].to(self.device)  # type: ignore[union-attr]

            # Get distilbert hidden states
            distilbert_output = self.model.distilbert(  # type: ignore[attr-defined]
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_state = distilbert_output[0]  # (batch, seq_len, dim)
            cls_output = hidden_state[:, 0]  # [CLS] token

            # Pass through pre_classifier + relu (penultimate)
            pre_classifier: nn.Linear = self.model.pre_classifier  # type: ignore[attr-defined]
            embeddings = torch.relu(pre_classifier(cls_output))
            all_embeddings.append(embeddings.float().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def get_grad_embeddings(
        self,
        dataset: TextClassificationDataset,
        batch_size: int = 64,
    ) -> NDArray[np.floating[Any]]:
        """Compute BADGE gradient embeddings.

        For each sample x with predicted label ŷ:
            g_c(x) = embedding(x) * (1[c == ŷ] - p_c(x))
        Concatenated across classes → shape (n_samples, embedding_dim * n_classes).
        """
        self.model.eval()
        n_classes = self.num_labels
        emb_dim = self.embedding_dim
        all_grad_embs: list[NDArray[np.floating[Any]]] = []
        loader = self._make_loader(dataset, batch_size)

        for batch in tqdm(loader, desc="BADGE grad embeddings", leave=False):
            input_ids = batch["input_ids"].to(self.device)  # type: ignore[union-attr]
            attention_mask = batch["attention_mask"].to(self.device)  # type: ignore[union-attr]

            with torch.no_grad():
                # Get embeddings (penultimate)
                distilbert_output = self.model.distilbert(  # type: ignore[attr-defined]
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_state = distilbert_output[0]
                cls_output = hidden_state[:, 0]
                pre_classifier: nn.Linear = self.model.pre_classifier  # type: ignore[attr-defined]
                embeddings = torch.relu(pre_classifier(cls_output))  # (batch, dim)

                # Get predictions
                logits = self.model.classifier(self.model.dropout(embeddings))  # type: ignore[attr-defined]
                probs = torch.softmax(logits.float(), dim=-1)  # (batch, n_classes)
                predicted = probs.argmax(dim=-1)  # (batch,)

            # Construct gradient embeddings: g_c = emb * (1[c==y_hat] - p_c)
            bsz = embeddings.shape[0]
            one_hot = torch.zeros(bsz, n_classes, device=self.device)
            one_hot.scatter_(1, predicted.unsqueeze(1), 1.0)

            # diff: (batch, n_classes), embeddings: (batch, dim)
            diff = one_hot - probs  # (batch, n_classes)
            # Outer product: (batch, n_classes, 1) * (batch, 1, dim) → (batch, n_classes, dim)
            grad_emb = diff.unsqueeze(2) * embeddings.unsqueeze(1)
            grad_emb = grad_emb.reshape(bsz, n_classes * emb_dim)  # (batch, n_classes * dim)
            all_grad_embs.append(grad_emb.float().cpu().numpy())

        return np.concatenate(all_grad_embs, axis=0)


def _enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
