"""Plotting utilities for FedGRAA reproduction — Figures 1, 2, and 3.

Gap 3.1 fix: the core library previously produced only stdout text; this module
provides matplotlib-based figure generation matching the paper's layouts.

Matplotlib is an optional dependency. A clear ``ImportError`` is raised if it
is not installed::

    pip install matplotlib

Typical usage::

    from fedgraa.experiment import run_gamma_sweep
    from fedgraa.plot import plot_gamma_sweep, plot_dual_metric

    results = run_gamma_sweep("CIC-IDS-2017", data_dir="data/CIC-IDS-2017")
    fig = plot_dual_metric(results, title="CIC-IDS-2017 — Figure 1")
    fig.savefig("figure1.png", dpi=150)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

try:
    import matplotlib.pyplot as _plt

    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False


def _require_mpl() -> "plt":
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting.\n"
            "Install it with:  pip install matplotlib"
        )
    return _plt


# Consistent style mapping shared by all plot functions.
_METHOD_STYLE = {
    "fedavg":        ("FedAvg",           "C0", "-",  "o"),
    "static":        ("Static Reparam.",  "C1", "--", "s"),
    "fedgraa_g0.0":  ("FedGRAA (γ=0)",   "C2", "-.", "^"),
    "fedgraa_g0.5":  ("FedGRAA (γ=0.5)", "C3", ":",  "D"),
    "fedgraa_g1.0":  ("FedGRAA (γ=1)",   "C4", "-",  "v"),
    "fedgraa":       ("FedGRAA (γ=1)",   "C4", "-",  "v"),
}


def plot_gamma_sweep(
    results: dict,
    metric: str = "last_n",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Figure":
    """Figure 1 style: accuracy vs. communication round for each method / γ.

    Parameters
    ----------
    results:
        Output of :func:`~fedgraa.experiment.run_gamma_sweep`.
    metric:
        ``"last_n"`` for rare-class accuracy (the paper's key metric) or
        ``"all"`` for overall accuracy.
    title:
        Figure title; auto-generated if *None*.
    save_path:
        If given, save the figure to this path at 150 dpi.
    ax:
        Optional pre-existing ``Axes`` to draw into (useful for subplots).
    """
    plt = _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    order = [
        "fedavg", "static", "fedgraa_g0.0", "fedgraa_g0.5", "fedgraa_g1.0"
    ]
    for key in order:
        if key not in results:
            continue
        label, color, ls, marker = _METHOD_STYLE[key]
        h = results[key].history
        y_vals = h.last_n_acc if metric == "last_n" else h.all_acc
        ax.plot(
            h.rounds, [v * 100 for v in y_vals],
            label=label, color=color, linestyle=ls, marker=marker, ms=4,
        )

    ylabel = (
        "Rare-class accuracy (%)" if metric == "last_n" else "Overall accuracy (%)"
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"γ sweep — {metric} accuracy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_dual_metric(
    results: dict,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Two-panel figure: overall accuracy (left) and rare-class accuracy (right).

    Reproduces the two-column layout of the paper's Figure 1 table as a line
    plot, showing all methods at once.
    """
    plt = _require_mpl()
    fig, (ax_all, ax_rare) = plt.subplots(1, 2, figsize=(14, 5))
    plot_gamma_sweep(results, metric="all",    ax=ax_all,  title="All-class accuracy")
    plot_gamma_sweep(results, metric="last_n", ax=ax_rare, title="Rare-class accuracy")
    fig.suptitle(title or "Figure 1 — γ sweep: all-class and rare-class accuracy")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_n_sweep(
    results: Dict[int, Dict[str, object]],
    metric: str = "last_n",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Figures 2-3 style: one panel per *n*, comparing FedAvg / Static / FedGRAA.

    Parameters
    ----------
    results:
        Output of :func:`~fedgraa.experiment.run_n_sweep`.
    metric:
        ``"last_n"`` or ``"all"``.
    """
    plt = _require_mpl()
    ns = sorted(results.keys())
    fig, axes = plt.subplots(1, len(ns), figsize=(6 * len(ns), 5), sharey=True)
    if len(ns) == 1:
        axes = [axes]

    for ax, n in zip(axes, ns):
        for key in ("fedavg", "static", "fedgraa"):
            if key not in results[n]:
                continue
            label, color, ls, marker = _METHOD_STYLE[key]
            h = results[n][key].history
            y_vals = h.last_n_acc if metric == "last_n" else h.all_acc
            ax.plot(
                h.rounds, [v * 100 for v in y_vals],
                label=label, color=color, linestyle=ls, marker=marker, ms=4,
            )
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Communication round")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    ylabel = (
        "Rare-class accuracy (%)" if metric == "last_n" else "Overall accuracy (%)"
    )
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title or f"Figures 2-3 — n sweep ({metric} accuracy)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_weight_evolution(
    history,
    client_labels: Optional[Sequence[str]] = None,
    top_k: int = 5,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Plot how the FedGRAA aggregation weights evolve over training rounds.

    Note: the strict reproduction does not apply a simplex projection, so
    weights may leave the probability simplex if Algorithm 2 drives them there.
    """
    plt = _require_mpl()
    if not history.weights:
        raise ValueError(
            "history.weights is empty. Run train() with eval_every < rounds."
        )

    weights = history.weights          # List[np.ndarray], one per eval round
    final_w = weights[-1]
    top_idx = np.argsort(final_w)[-top_k:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- left: per-client weight trajectories ---
    ax = axes[0]
    for k in top_idx:
        label = (
            client_labels[k] if client_labels and k < len(client_labels)
            else f"Client {k}"
        )
        ax.plot(
            history.rounds, [w[k] for w in weights], label=label, marker="o", ms=3,
        )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Aggregation weight $w_k$")
    ax.set_title(f"Top-{top_k} client weight trajectories")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- right: simplex sanity checks (sum and min) ---
    ax2 = axes[1]
    sums = [w.sum() for w in weights]
    mins  = [w.min() for w in weights]
    ax2.plot(history.rounds, sums,  label="sum(w)", color="C0", marker="o", ms=3)
    ax2.plot(history.rounds, mins,  label="min(w)", color="C1", marker="s", ms=3)
    ax2.axhline(1.0, color="C0", lw=0.8, linestyle="--", alpha=0.5)
    ax2.axhline(0.0, color="C1", lw=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Communication round")
    ax2.set_ylabel("Value")
    ax2.set_title("Simplex constraints: sum(w)=1 and min(w)≥0")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title or "FedGRAA weight evolution")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_composite_loss(
    histories: Dict[str, object],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Plot L'(w^t) vs round for one or more FedGRAA runs.

    Useful for verifying Lemma 1 (non-increasing composite loss, gap 3.7).
    """
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, h in histories.items():
        finite_rounds = [
            r for r, l in zip(h.rounds, h.composite_loss) if np.isfinite(l)
        ]
        finite_loss = [l for l in h.composite_loss if np.isfinite(l)]
        if finite_rounds:
            ax.plot(finite_rounds, finite_loss, label=label, marker="o", ms=3)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Composite loss $L'(w^t)$")
    ax.set_title(title or "Composite loss vs. round (Lemma 1 check)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
