from __future__ import annotations

from pathlib import Path

import numpy as np


def confusion_figure(M: np.ndarray, labels: list[str], out: Path, title: str | None = None,
                     tick_step: int | None = None, figsize=(5.2, 4.2)) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.asarray(M)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1))
    cb = fig.colorbar(im, ax=ax)
    if tick_step:
        cb.set_ticks(range(0, int(M.max()) + 1, tick_step))
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0 if len(labels) <= 4 else 45, ha="center" if len(labels) <= 4 else "left")
    ax.set_yticklabels(labels)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel("Prediction", fontsize=11); ax.set_ylabel("Actual", fontsize=11)
    if title:
        ax.set_title(title, fontsize=15, pad=15)
    thr = M.max() / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(int(M[i, j])), ha="center", va="center",
                    color="white" if M[i, j] > thr else "#08306b", fontsize=10 if len(labels) <= 4 else 7)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
