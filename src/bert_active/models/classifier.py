"""Classifier wrapper with embedding and gradient extraction for AL strategies.

Supports both DistilBERT and BERT (including DNABERT-2) architectures.
"""

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
    """Create a model for sequence classification."""
    dtype_map: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=dtype_map.get(torch_dtype, torch.float16),
        trust_remote_code=True,
    )
    return model


class ModelWrapper:
    """Wraps a classifier to provide inference utilities needed by AL strategies.

    Automatically detects DistilBERT vs BERT architecture and adapts accordingly.

    Provides:
    - predict_proba: softmax probabilities
    - predict_proba_dropout: MC Dropout probabilities
    - get_embeddings: penultimate layer embeddings
    - get_grad_embeddings: gradient embeddings for BADGE
    """

    def __init__(self, model: PreTrainedModel, device: str = "cuda") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model = self.model.to(self.device)  # type: ignore[assignment]

        # Detect architecture
        self._is_distilbert = hasattr(self.model, "distilbert")

    @property
    def num_labels(self) -> int:
        return self.model.config.num_labels  # type: ignore[no-any-return]

    @property
    def embedding_dim(self) -> int:
        if self._is_distilbert:
            return self.model.config.dim  # type: ignore[no-any-return]
        return self.model.config.hidden_size  # type: ignore[no-any-return]

    def _get_backbone(self) -> nn.Module:
        """Get the transformer backbone (distilbert or bert)."""
        if self._is_distilbert:
            return self.model.distilbert  # type: ignore[attr-defined]
        if hasattr(self.model, "bert"):
            return self.model.bert  # type: ignore[attr-defined]
        # Fallback: try common attribute names
        for attr in ("roberta", "electra", "albert"):
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        msg = f"Cannot find backbone in model: {type(self.model)}"
        raise AttributeError(msg)

    def _get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get [CLS] token embedding from penultimate layer.

        For DistilBERT: hidden[:, 0] -> pre_classifier -> ReLU
        For BERT: hidden[:, 0] -> pooler (Linear + Tanh), or just CLS if no pooler
        """
        backbone = self._get_backbone()
        output = backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]  # (batch, seq_len, dim)
        cls_output = hidden_state[:, 0]  # [CLS] token

        if self._is_distilbert:
            pre_classifier: nn.Linear = self.model.pre_classifier  # type: ignore[attr-defined]
            return torch.relu(pre_classifier(cls_output))

        # BERT: use pooler if available
        if hasattr(self.model, "bert") and hasattr(self.model.bert, "pooler") and self.model.bert.pooler is not None:  # type: ignore[attr-defined]
            return self.model.bert.pooler(hidden_state)  # type: ignore[attr-defined]
        return cls_output

    def _get_logits_from_embedding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get logits from embeddings through the classification head."""
        if self._is_distilbert:
            x = self.model.dropout(embeddings)  # type: ignore[attr-defined]
            return self.model.classifier(x)  # type: ignore[attr-defined]

        # BERT: dropout -> classifier
        dropout = self.model.dropout  # type: ignore[attr-defined]
        classifier = self.model.classifier  # type: ignore[attr-defined]
        return classifier(dropout(embeddings))

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
        """Extract penultimate layer embeddings."""
        self.model.eval()
        all_embeddings: list[NDArray[np.floating[Any]]] = []
        loader = self._make_loader(dataset, batch_size)

        for batch in tqdm(loader, desc="Extracting embeddings", leave=False):
            input_ids = batch["input_ids"].to(self.device)  # type: ignore[union-attr]
            attention_mask = batch["attention_mask"].to(self.device)  # type: ignore[union-attr]
            embeddings = self._get_cls_embedding(input_ids, attention_mask)
            all_embeddings.append(embeddings.float().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def get_grad_embeddings(
        self,
        dataset: TextClassificationDataset,
        batch_size: int = 64,
    ) -> NDArray[np.floating[Any]]:
        """Compute BADGE gradient embeddings.

        For each sample x with predicted label y_hat:
            g_c(x) = embedding(x) * (1[c == y_hat] - p_c(x))
        Concatenated across classes -> shape (n_samples, embedding_dim * n_classes).
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
                embeddings = self._get_cls_embedding(input_ids, attention_mask)
                logits = self._get_logits_from_embedding(embeddings)
                probs = torch.softmax(logits.float(), dim=-1)
                predicted = probs.argmax(dim=-1)

            # Construct gradient embeddings: g_c = emb * (1[c==y_hat] - p_c)
            bsz = embeddings.shape[0]
            one_hot = torch.zeros(bsz, n_classes, device=self.device)
            one_hot.scatter_(1, predicted.unsqueeze(1), 1.0)

            diff = one_hot - probs
            grad_emb = diff.unsqueeze(2) * embeddings.unsqueeze(1)
            grad_emb = grad_emb.reshape(bsz, n_classes * emb_dim)
            all_grad_embs.append(grad_emb.float().cpu().numpy())

        return np.concatenate(all_grad_embs, axis=0)


def _enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
