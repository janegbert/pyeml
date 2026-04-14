"""Multi-phase training engine for EML symbolic regression."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
import torch

from pyeml._config import TrainConfig, SearchConfig
from pyeml._operator import DTYPE
from pyeml._tree import EMLTree
from pyeml._symbolic import extract_expression, simplify

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SearchResult:
    """Result of an EML symbolic regression search."""

    model: EMLTree
    eml_expression: str
    expression: str  # simplified
    snapped_loss: float
    depth: int
    is_exact: bool
    n_restarts_tried: int
    train_history: list[float] = field(default_factory=list)


def search(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    config: SearchConfig | None = None,
) -> SearchResult | None:
    """Run multi-depth EML symbolic regression.

    Tries increasing depths and multiple random restarts at each depth.
    Returns the best result, or None if all attempts failed.
    """
    if config is None:
        config = SearchConfig()

    # Convert inputs to complex128 tensors
    if isinstance(x, np.ndarray):
        x_t = torch.tensor(x, dtype=DTYPE)
    else:
        x_t = x.to(DTYPE)

    if isinstance(y, np.ndarray):
        y_t = torch.tensor(y, dtype=DTYPE)
    else:
        y_t = y.to(DTYPE)

    best: SearchResult | None = None

    for depth in config.depths:
        if config.verbose:
            n_params = 5 * 2**depth - 6
            print(f"\n  Depth {depth} ({n_params} params, {config.n_restarts} restarts)")

        result = _search_depth(x_t, y_t, depth, config)

        if result is not None:
            if best is None or result.snapped_loss < best.snapped_loss:
                best = result

            if result.is_exact:
                if config.verbose:
                    print(f"  >>> EXACT match at depth {depth}: {result.expression}")
                return best

    return best


def _search_depth(
    x: torch.Tensor,
    y: torch.Tensor,
    depth: int,
    config: SearchConfig,
) -> SearchResult | None:
    """Run multiple restarts at a single depth."""
    tc = config.train
    best_model = None
    best_loss = float("inf")
    best_history: list[float] = []
    total_tried = 0

    for run in range(config.n_restarts):
        total_tried += 1
        model = EMLTree(depth=depth)
        result = _train_single(model, x, y, tc)

        if result is None:
            if config.verbose:
                print(f"    Run {run+1}: DIVERGED")
            continue

        loss, history = result

        # Snap and evaluate
        model.snap_weights()
        with torch.no_grad():
            y_pred = model(x)
            diff = y_pred - y
            snapped_loss = torch.mean(diff.real**2 + diff.imag**2).item()

        is_exact = snapped_loss < config.exact_threshold

        if config.verbose:
            tag = "EXACT" if is_exact else ("CLOSE" if snapped_loss < 0.01 else "")
            print(f"    Run {run+1}: loss={loss:.2e} -> snapped={snapped_loss:.2e} {tag}")

        if snapped_loss < best_loss:
            best_loss = snapped_loss
            best_model = model
            best_history = history

        if is_exact:
            break

    if best_model is None:
        return None

    eml_expr = extract_expression(best_model)
    return SearchResult(
        model=best_model,
        eml_expression=eml_expr,
        expression=simplify(eml_expr),
        snapped_loss=best_loss,
        depth=depth,
        is_exact=best_loss < config.exact_threshold,
        n_restarts_tried=total_tried,
        train_history=best_history,
    )


def _train_single(
    model: EMLTree,
    x: torch.Tensor,
    y: torch.Tensor,
    tc: TrainConfig,
) -> tuple[float, list[float]] | None:
    """Train one model. Returns (final_loss, history) or None if diverged."""
    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr)
    harden_epoch = int(tc.epochs * tc.harden_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=harden_epoch)

    history: list[float] = []

    for epoch in range(tc.epochs):
        optimizer.zero_grad()

        y_pred = model(x, gumbel=tc.use_gumbel)

        diff = y_pred - y
        loss = torch.mean(diff.real**2 + diff.imag**2)

        if torch.isnan(loss) or torch.isinf(loss):
            return None

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()

        if epoch < harden_epoch:
            scheduler.step()

        # Hardening phase
        if epoch >= harden_epoch:
            progress = (epoch - harden_epoch) / max(1, tc.epochs - harden_epoch)
            factor = tc.harden_base + tc.harden_growth * progress
            model.harden(factor)

        loss_val = loss.item()
        history.append(loss_val)

        if loss_val < tc.convergence_threshold:
            break

    return loss_val, history
