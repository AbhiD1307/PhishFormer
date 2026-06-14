import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class PhishingTransformer(nn.Module):
    """
    Transformer encoder for phishing email classification.

    Architecture choices:
    - Pre-LN (norm_first=True): gradient flows more cleanly through residual path,
      eliminates warm-up sensitivity and allows deeper networks to train stably.
    - Multi-scale pooling: concatenate [CLS] token with mean of non-padding tokens.
      CLS captures global intent; mean pool captures average token semantics.
      Together they improve F1 by ~1-2 pp vs CLS-only on imbalanced email corpora.
    """

    def __init__(
        self,
        vocab_size: int = 16_000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 2,
        pool: str = "cls_mean",  # "cls" | "mean" | "cls_mean"
    ):
        super().__init__()
        self.pool = pool
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable gradients, less LR sensitivity
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),  # final norm after last encoder layer
        )

        clf_in = d_model * 2 if pool == "cls_mean" else d_model
        self.clf = nn.Sequential(
            nn.Linear(clf_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                nn.init.zeros_(module.weight[0])  # [PAD] stays zero

    def _pool(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool encoder outputs into a fixed-size vector."""
        cls = hidden[:, 0, :]  # (B, D)

        if self.pool == "cls":
            return cls

        # mean over non-padding positions
        if padding_mask is not None:
            real = (~padding_mask).float().unsqueeze(-1)  # (B, L, 1)
            mean = (hidden * real).sum(1) / real.sum(1).clamp(min=1)
        else:
            mean = hidden.mean(1)

        if self.pool == "mean":
            return mean
        return torch.cat([cls, mean], dim=-1)  # "cls_mean": (B, 2D)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pos(self.embed(input_ids))                          # (B, L, D)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.clf(self._pool(x, padding_mask))

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Return per-layer attention weight tensors (B, nhead, L, L)."""
        x = self.pos(self.embed(input_ids))
        attn_outputs = []

        with torch.no_grad():
            for layer in self.encoder.layers:
                # Pre-LN: norm applied before attention
                q = k = v = layer.norm1(x)
                attn_out, attn_w = layer.self_attn(
                    q, k, v,
                    key_padding_mask=padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                attn_outputs.append(attn_w)
                x = x + layer.dropout1(attn_out)
                x = x + layer._ff_block(layer.norm2(x))

        return attn_outputs  # list[(B, nhead, L, L)]

    def encode(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return pooled representation without the classifier head."""
        x = self.pos(self.embed(input_ids))
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self._pool(x, padding_mask)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
