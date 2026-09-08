"""Bounded-memory Contriever mean pooling on CPU, Apple MPS or CUDA."""

import numpy as np
import torch
from hipporag.embedding_model.base import BaseEmbeddingModel
from transformers import AutoModel, AutoTokenizer


class LocalContriever(BaseEmbeddingModel):
    """Use explicit placement instead of upstream's automatic device map."""

    def __init__(self, global_config, device):
        super().__init__(global_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, trust_remote_code=False)
        self.embedding_model = (
            AutoModel.from_pretrained(
                self.embedding_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=False,
            )
            .to(device)
            .eval()
        )
        self.embedding_dim = self.embedding_model.config.hidden_size

    def batch_encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        batch_size = kwargs.get("batch_size", self.global_config.embedding_batch_size)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size 必须是正整数")
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.global_config.embedding_max_seq_len,
                    return_tensors="pt",
                ).to(self.embedding_model.device)
                hidden = self.embedding_model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).bool()
                pooled = hidden.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                # Transfer each batch immediately so GPU/MPS memory doesn't grow with corpus size.
                batches.append(pooled.cpu().numpy())
        return self._normalize_embeddings(np.concatenate(batches))

    def close(self):
        self.embedding_model = None
        self.tokenizer = None
