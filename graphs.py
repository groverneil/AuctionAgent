"""Graphing: training curves and eval bar chart."""

import os
from typing import Dict, Any, List

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def smooth(y: list, window: int = 50) -> np.ndarray:
    """Moving average smoothing."""
    if len(y) < window:
        return np.array(y)
    return np.convolve(y, np.ones(window) / window, mode="valid")


def plot_training(
    history: Dict[str, Any],
    out_dir: str = ".",
    save: bool = True,
    show: bool = True,
) -> None:
    """Plot reward and loss over episodes."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    rewards = history.get("episode_reward", [])
    losses = history.get("episode_loss", [])
    if not rewards:
        print("No training data to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(rewards, alpha=0.3, color="steelblue", linewidth=0.5)
    if len(rewards) >= 50:
        smoothed = smooth(rewards, 50)
        ax.plot(range(50 - 1, len(rewards)), smoothed, color="steelblue", linewidth=2, label="Smoothed (50-ep)")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training: Reward over Episodes")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if losses:
        ax.plot(losses, alpha=0.3, color="coral", linewidth=0.5)
        if len(losses) >= 50:
            smoothed = smooth(losses, 50)
            ax.plot(range(50 - 1, len(losses)), smoothed, color="coral", linewidth=2, label="Smoothed (50-ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("Training: Policy Loss over Episodes")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
        print(f"Saved {out_dir}/training_curves.png")
    if show:
        plt.show()
    else:
        plt.close()


def plot_eval(
    all_wins: Dict[str, List[float]],
    n_rounds: int,
    agent_heuristic: Dict[str, str],
    out_dir: str = ".",
    save: bool = True,
    show: bool = True,
) -> None:
    """Plot evaluation bar chart: wins per agent."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    rankings = sorted(all_wins.keys(), key=lambda n: (-np.mean(all_wins[n]), n))
    names = []
    means = []
    stds = []
    for name in rankings:
        wins = all_wins[name]
        suffix = f" ({agent_heuristic[name]})" if name in agent_heuristic else ""
        names.append(f"{name}{suffix}"[:40])
        means.append(np.mean(wins))
        stds.append(np.std(wins))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    ax.barh(x, means, xerr=stds, capsize=3, color="steelblue", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Rounds Won per Auction")
    ax.set_title(f"Evaluation: Agent Performance ({n_rounds} rounds/auction)")
    ax.axvline(n_rounds / len(names), color="gray", linestyle="--", alpha=0.5, label="Equal share")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(out_dir, "eval_results.png"), dpi=150, bbox_inches="tight")
        print(f"Saved {out_dir}/eval_results.png")
    if show:
        plt.show()
    else:
        plt.close()
