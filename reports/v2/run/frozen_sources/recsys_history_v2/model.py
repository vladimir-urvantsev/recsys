"""Candidate-conditioned event-history network for the V2 response task.

The Transformer sees an already availability-filtered user-day history once.
Candidates only cross-attend to that representation; they never attend to one
another. This is a DIN/BST-inspired architecture, not an implementation of
SASRec's next-item objective or of HSTU. Raw IDs use two independent prime-mixed
hash banks, so new IDs do not require fitting a vocabulary on future data.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PAD_ID = -2
MISSING_ID = -1
DEFAULT_BUCKETS = {"item": 131072, "artist": 32768, "album": 131072, "user": 8192}
ENTITY_DIMENSIONS = {"item": 32, "artist": 16, "album": 16, "user": 16}


class DualHashEmbedding(nn.Module):
    """Two summed hash embeddings with reserved PAD and missing buckets.

    Mixing modulo distinct large primes *before* the bucket modulus avoids
    the identical collision classes of two affine hashes modulo the same power
    of two. For source IDs <1e9, multiplication remains within signed int64.
    PAD=0 has a fixed zero embedding; missing=1 is a learned distinct token.
    """

    _HASHES = ((2654435761, 1013904223, 2147483647),
               (2246822519, 1664525, 2147483629))

    def __init__(self, buckets: int, dimension: int):
        super().__init__()
        if int(buckets) < 2:
            raise ValueError("Each hash bank requires at least two known-ID buckets")
        self.buckets = int(buckets)
        self.dimension = int(dimension)
        self.tables = nn.ModuleList([
            nn.Embedding(self.buckets + 2, self.dimension, padding_idx=0)
            for _ in range(2)
        ])
        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                table.weight[0].zero_()

    def bucket_indices(self, raw_ids: Tensor, bank: int) -> Tensor:
        if bank not in (0, 1):
            raise ValueError("Hash bank must be 0 or 1")
        ids = raw_ids.to(dtype=torch.long)
        multiplier, salt, prime = self._HASHES[bank]
        # Reducing first also protects hashing if a caller supplies an ID beyond
        # the source's range, without changing the defined hash for valid IDs.
        positive = ids.clamp_min(0).remainder(prime)
        mixed = (positive * multiplier + salt).remainder(prime)
        known = mixed.remainder(self.buckets) + 2
        return torch.where(ids == PAD_ID, torch.zeros_like(known),
                           torch.where(ids < 0, torch.ones_like(known), known))

    def forward(self, raw_ids: Tensor) -> Tensor:
        return sum(table(self.bucket_indices(raw_ids, bank))
                   for bank, table in enumerate(self.tables))


def exact_relation_matches(candidate_ids: Tensor, history_ids: Tensor) -> Tensor:
    """Return exact known-ID matches (B,C,L,3), never missing/PAD matches."""
    candidate = candidate_ids.unsqueeze(-2)
    history = history_ids.unsqueeze(-3)
    return (candidate == history) & (candidate >= 0) & (history >= 0)


class EventHistoryModel(nn.Module):
    """Shared history encoder plus candidate-specific multi-head attention.

    ``forward`` returns raw logits (B,C); padded candidates are set to zero.
    The caller still applies ``candidate_mask`` to its loss and metrics.
    Current outcomes are absent from the explicit input contract. Historical
    ``history_age`` is already log1p(days since the experiment cutoff).

    With ``n_layers=0``, no positional embedding or inter-event mixing is used.
    Attention pooling still receives historical age, so this is an ablation of
    the inter-event encoder, not an ablation of all temporal information.
    """

    def __init__(self, config: dict[str, Any], global_dim: int = 37):
        super().__init__()
        self.config = dict(config)
        self.d_model = int(config.get("d_model", 128))
        self.n_layers = int(config.get("n_layers", 2))
        self.n_heads = int(config.get("n_heads", 4))
        self.dropout_probability = float(config.get("dropout", 0.1))
        self.global_dim = int(global_dim)
        self.max_context = int(config.get("context_recent", 256)) + int(config.get("context_older", 256))
        if self.d_model < 1 or self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.n_layers < 0 or self.global_dim < 0 or self.max_context < 1:
            raise ValueError("Invalid number of layers, global dimension, or context length")
        if not 0 <= self.dropout_probability < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.head_dimension = self.d_model // self.n_heads
        buckets = {**DEFAULT_BUCKETS, **config.get("embedding_buckets", {})}
        self.id_embeddings = nn.ModuleDict({
            key: DualHashEmbedding(buckets[key], dimension)
            for key, dimension in ENTITY_DIMENSIONS.items()
        })
        self.type_embedding = nn.Embedding(6, 8, padding_idx=0)
        # PAD=0, missing=1; historical ratios 0..200 in ten-percentage buckets
        # occupy 2..22. The cap is representational, never an availability cap.
        self.ratio_embedding = nn.Embedding(23, 8, padding_idx=0)
        self.organic_embedding = nn.Embedding(3, 4, padding_idx=0)
        self.age_projection = nn.Sequential(nn.Linear(1, 8), nn.SiLU())
        self.history_projection = nn.Sequential(nn.Linear(92, self.d_model), nn.LayerNorm(self.d_model))
        self.query_projection = nn.Sequential(nn.Linear(64, self.d_model), nn.LayerNorm(self.d_model))
        self.null_history = nn.Parameter(torch.empty(1, 1, self.d_model))
        nn.init.normal_(self.null_history, std=0.02)
        self.position_embedding = nn.Embedding(self.max_context, self.d_model) if self.n_layers else None
        # Separate constructors initialize different layers independently. A
        # ModuleList avoids TransformerEncoder's identical-clone initialization.
        self.history_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.n_heads,
                dim_feedforward=4 * self.d_model, dropout=self.dropout_probability,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(self.n_layers)
        ])
        self.memory_norm = nn.LayerNorm(self.d_model)
        self.query_attention = nn.Linear(self.d_model, self.d_model)
        self.key_attention = nn.Linear(self.d_model, self.d_model)
        self.value_attention = nn.Linear(self.d_model, self.d_model)
        self.context_projection = nn.Linear(self.d_model, self.d_model)
        self.exact_match_bias = nn.Parameter(torch.zeros(self.n_heads, 3))
        head_input = 4 * self.d_model + ENTITY_DIMENSIONS["user"] + self.global_dim
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(head_input), nn.Linear(head_input, 128), nn.SiLU(),
            nn.Dropout(self.dropout_probability), nn.Linear(128, 1),
        )

    def _entity_embedding(self, ids: Tensor) -> Tensor:
        return torch.cat([
            self.id_embeddings[key](ids[..., index])
            for index, key in enumerate(("item", "artist", "album"))
        ], dim=-1)

    def encode_history(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Encode once per user-day; returned mask includes the valid CLS token."""
        valid = batch["history_mask"].to(dtype=torch.bool)
        if valid.ndim != 2:
            raise ValueError("history_mask must have shape (B,L)")
        batch_size, length = valid.shape
        if length > self.max_context:
            raise ValueError(f"History length {length} exceeds max_context={self.max_context}")
        ids = batch["history_ids"].to(dtype=torch.long)
        if ids.shape != (batch_size, length, 3):
            raise ValueError("history_ids must have shape (B,L,3)")
        # Mask every payload before embedding/projection: even arbitrary padded
        # values cannot affect history state or exact raw-ID match gates.
        ids = torch.where(valid.unsqueeze(-1), ids, torch.full_like(ids, PAD_ID))
        event_type = torch.where(valid, batch["history_type"].long(), 0).clamp(0, 5)
        ratio = batch["history_ratio"].long()
        ratio_bucket = ratio.clamp(0, 200).div(10, rounding_mode="floor") + 2
        ratio_bucket = torch.where(ratio < 0, 1, ratio_bucket)
        ratio_bucket = torch.where(valid, ratio_bucket, 0)
        organic = torch.where(valid, batch["history_organic"].long().clamp(0, 1) + 1, 0)
        age = torch.where(valid, batch["history_age"], 0).to(dtype=self.null_history.dtype)
        token = self.history_projection(torch.cat([
            self._entity_embedding(ids), self.type_embedding(event_type),
            self.ratio_embedding(ratio_bucket), self.organic_embedding(organic),
            self.age_projection(age.unsqueeze(-1)),
        ], dim=-1))
        if self.position_embedding is not None:
            # Position depends on valid-event rank, not where PAD is placed.
            positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
            token = token + self.position_embedding(positions)
        token = token.masked_fill(~valid.unsqueeze(-1), 0)
        memory = torch.cat([self.null_history.expand(batch_size, -1, -1), token], dim=1)
        memory_valid = torch.cat([torch.ones((batch_size, 1), dtype=torch.bool, device=valid.device), valid], dim=1)
        for layer in self.history_layers:
            # All tokens are already past the same availability cutoff. There
            # is no next-action auxiliary target, so bidirectional history
            # attention is permitted; it never sees a current query outcome.
            memory = layer(memory, src_key_padding_mask=~memory_valid)
            memory = memory.masked_fill(~memory_valid.unsqueeze(-1), 0)
        return self.memory_norm(memory), memory_valid, ids

    def score_candidates(self, memory: Tensor, memory_valid: Tensor, history_ids: Tensor,
                         batch: dict[str, Tensor]) -> Tensor:
        """Score candidates independently against a previously encoded history."""
        valid_candidates = batch["candidate_mask"].to(dtype=torch.bool)
        ids = batch["candidate_ids"].to(dtype=torch.long)
        if ids.ndim != 3 or ids.shape[:2] != valid_candidates.shape or ids.shape[-1] != 3:
            raise ValueError("candidate_ids must have shape (B,C,3) matching candidate_mask")
        ids = torch.where(valid_candidates.unsqueeze(-1), ids, torch.full_like(ids, PAD_ID))
        query = self.query_projection(self._entity_embedding(ids))
        batch_size, n_candidates, _ = query.shape
        memory_length = memory.shape[1]
        split_heads = lambda value, length: value.reshape(batch_size, length, self.n_heads, self.head_dimension).transpose(1, 2)
        q = split_heads(self.query_attention(query), n_candidates)
        k = split_heads(self.key_attention(memory), memory_length)
        v = split_heads(self.value_attention(memory), memory_length)
        attention_logits = torch.matmul(q, k.transpose(-2, -1)).float() / math.sqrt(self.head_dimension)
        matches = exact_relation_matches(ids, history_ids).to(dtype=self.exact_match_bias.dtype)
        match_logits = torch.einsum("bclr,hr->bhcl", matches, self.exact_match_bias)
        attention_logits = attention_logits + F.pad(match_logits.float(), (1, 0))
        attention_logits = attention_logits.masked_fill(~memory_valid[:, None, None, :], float("-inf"))
        attention = torch.softmax(attention_logits, dim=-1).to(dtype=v.dtype)
        attention = F.dropout(attention, p=self.dropout_probability, training=self.training)
        context = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size, n_candidates, self.d_model)
        context = self.context_projection(context)
        uid = self.id_embeddings["user"](batch["group_uid"].long())[:, None, :].expand(-1, n_candidates, -1)
        components = [query, context, query * context, torch.abs(query - context), uid]
        if self.global_dim:
            global_features = batch["global_features"]
            if global_features.shape != (batch_size, n_candidates, self.global_dim):
                raise ValueError(f"global_features must have final dimension {self.global_dim}")
            global_features = torch.where(valid_candidates.unsqueeze(-1), global_features, 0)
            components.append(global_features.to(dtype=query.dtype))
        logits = self.prediction_head(torch.cat(components, dim=-1)).squeeze(-1)
        return logits.masked_fill(~valid_candidates, 0)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.score_candidates(*self.encode_history(batch), batch)

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if not trainable_only or parameter.requires_grad)

    def description(self) -> dict[str, Any]:
        return {"architecture": "shared user-day event encoder plus independent candidate cross-attention",
                "d_model": self.d_model, "n_layers": self.n_layers, "n_heads": self.n_heads,
                "max_context": self.max_context, "global_dim": self.global_dim,
                "parameter_count": self.num_parameters(),
                "trainable_parameter_count": self.num_parameters(trainable_only=True),
                "hash_buckets": {name: embedding.buckets for name, embedding in self.id_embeddings.items()},
                "historical_ratio_embedding": "10-percentage buckets, capped at 200%; missing separate",
                "history_attention": "bidirectional over already available past only",
                "no_event_encoder": self.n_layers == 0}
