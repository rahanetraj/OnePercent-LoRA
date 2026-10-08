"""Implémentation « from scratch » de LoRA (Hu et al., 2021).

Pour une couche linéaire gelée W0 ∈ R^{d_out × d_in}, LoRA apprend une mise à
jour de rang faible ΔW = B A, avec A ∈ R^{r × d_in}, B ∈ R^{d_out × r}, r ≪ min(d_in, d_out) :

    h = W0 x + (alpha / r) · B A x

B est initialisée à zéro, donc ΔW = 0 au départ : le modèle commence exactement
comme le modèle pré-entraîné.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Enveloppe une ``nn.Linear`` gelée et y ajoute une branche de rang ``r``."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("Le rang r doit être strictement positif")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        # (x A^T) B^T : on ne matérialise jamais la matrice d_out × d_in
        return out + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """W ← W0 + (alpha/r) B A : aucune latence supplémentaire à l'inférence."""
        if not self.merged:
            self.base.weight += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def extra_repr(self) -> str:
        return f"in={self.base.in_features}, out={self.base.out_features}, r={self.r}, alpha={self.alpha}"


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    trainable_modules: Iterable[str] = (),
) -> nn.Module:
    """Gèle tout le modèle puis remplace les ``nn.Linear`` ciblées par des ``LoRALinear``.

    ``trainable_modules`` : noms de sous-modules laissés entraînables (ex. la tête de classification).
    """
    targets = tuple(target_modules)
    keep = tuple(trainable_modules)
    for p in model.parameters():
        p.requires_grad_(False)

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))

    for name, p in model.named_parameters():
        if any(k in name for k in keep):
            p.requires_grad_(True)
    return model


def merge_all(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
