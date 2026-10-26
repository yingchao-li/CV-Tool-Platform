"""Ablation experiments for the VLAD retrieval pipeline."""

from __future__ import annotations

EXPERIMENTS = ("intranorm", "pca", "aqe", "codebook", "inversion")

__all__ = ["EXPERIMENTS"]
