"""Configuration dataclasses for EML training and search."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Configuration for a single training run."""

    lr: float = 0.01
    epochs: int = 5000
    harden_start: float = 0.8  # fraction of epochs where hardening begins
    harden_base: float = 1.001
    harden_growth: float = 0.004  # factor grows as base + growth * progress
    grad_clip: float = 5.0
    convergence_threshold: float = 1e-28
    use_gumbel: bool = False
    gumbel_tau_start: float = 1.0
    gumbel_tau_end: float = 0.1


@dataclass
class SearchConfig:
    """Configuration for multi-depth symbolic search."""

    depths: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    n_restarts: int = 10
    n_points: int = 200
    x_range: tuple[float, float] = (0.5, 3.0)
    exact_threshold: float = 1e-20
    verbose: bool = True
    train: TrainConfig = field(default_factory=TrainConfig)
