import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent
EVAL = BASE / "evaluation"
FIG  = EVAL / "figures"
FIG.mkdir(exist_ok=True)

VARIANTS   = ["baseline", "exact", "phonetic", "llm_corr",
              "placeholder", "placeholder_ext",
              "claude_haiku", "claude", "gemini", "llm_mt"]
VAR_LABELS = ["Baseline\nOPUS-MT",
              "Exact-match\nкорекція ASR",
              "Фонетична\nкорекція ASR",
              "LLM-корекція\nASR",
              "Code-switching\n(52 dict)",
              "Code-switching\n(176 dict)",
              "Claude\nHaiku 4.5",
              "Claude\nSonnet 4",
              "Gemini 2.0\nFlash",
              "GPT-4o"]
COLORS     = ["#999999", "#4C78A8", "#72B7B2", "#54A24B",
              "#F58518", "#E45756",
              "#D4A5C8", "#B279A2", "#8E6698", "#6A4E7C"]


def load_results():
    with open(EVAL / "intervention_results.json", encoding="utf-8") as f:
        inter = json.load(f)
    with open(EVAL / "intervention_entity_summary.json", encoding="utf-8") as f:
        entity = json.load(f)
    with open(EVAL / "placeholder_results.json", encoding="utf-8") as f:
        repair = json.load(f)
    return inter, entity, repair


def fig_bars(inter, entity):
    """Зведені стовпчики: 6 варіантів × 4 метрики."""
    metrics = [
        ("bleu",    "BLEU",       0, 65),
        ("bert_f1", "BERTScore-F1", 0, 0.8),
        ("meteor",  "METEOR",     0, 0.8),
        ("f1",      "Entity F1",  0, 0.9),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (key, title, ymin, ymax) in zip(axes.flat, metrics):
        values = []
        for var in VARIANTS:
            if key == "f1":
                values.append(entity[var]["f1_mean"])
            else:
                values.append(inter["summary"][var][key])

        bars = ax.bar(range(len(VARIANTS)), values, color=COLORS,
                       edgecolor="black", alpha=0.85)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(VAR_LABELS, fontsize=9, rotation=20, ha="right")
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ymax - ymin) * 0.015,
                    f"{val:.2f}" if val > 1 else f"{val:.3f}",
                    ha="center", fontsize=9)

        ax.axhline(y=values[0], color="gray", ls="--", lw=0.7, alpha=0.5)

    plt.suptitle("Порівняння варіантів MT-pipeline за автоматичними метриками (n = 12 сегментів)",
                  fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(FIG / "intervention_bars.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("saved: intervention_bars.png")


def fig_entity_heatmap():
    """Heatmap Entity F1: сегмент × варіант."""
    df = pd.read_csv(EVAL / "intervention_entity_f1.csv")
    pivot = df.pivot(index="segment", columns="variant", values="f1")
    pivot = pivot[VARIANTS]

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(VAR_LABELS, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([s.replace("_", " ") for s in pivot.index], fontsize=9)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            color = "white" if v < 0.4 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Entity F1")
    ax.set_title("Entity F1 по сегментах та варіантах")
    plt.tight_layout()
    plt.savefig(FIG / "intervention_entity_heatmap.png", dpi=140,
                bbox_inches="tight")
    plt.close()
    print("saved: intervention_entity_heatmap.png")


def fig_survival(repair):
    """Survival rate: pre-repair vs post-repair."""
    per_seg = repair["per_segment"]
    segments = [r["segment"].replace("_", " ") for r in per_seg]
    pre  = [r["survival_pre"]  * 100 for r in per_seg]
    post = [r["survival_post"] * 100 for r in per_seg]

    n_entities = [r["n_entities_found"] for r in per_seg]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [3, 2]})

    x = np.arange(len(segments))
    width = 0.38
    ax1.bar(x - width / 2, pre,  width, label="До repair",
             color="#E45756", edgecolor="black", alpha=0.8)
    ax1.bar(x + width / 2, post, width, label="Після repair",
             color="#54A24B", edgecolor="black", alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(segments, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Survival rate (%)")
    ax1.set_title("Виживання EN-сутностей у перекладі OPUS-MT\nдо та після post-MT repair")
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    for i, n in enumerate(n_entities):
        ax1.text(i, 103, f"n={n}", ha="center", fontsize=7, color="gray")

    overall_pre  = repair["survival_rate_pre"]  * 100
    overall_post = repair["survival_rate_post"] * 100
    ax2.bar([0, 1], [overall_pre, overall_post],
             color=["#E45756", "#54A24B"], edgecolor="black",
             width=0.5, alpha=0.8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["До repair", "Після repair"], fontsize=11)
    ax2.set_ylabel("Survival rate (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title(f"Усього: {repair['total_entities']} сутностей")
    for i, v in enumerate([overall_pre, overall_post]):
        ax2.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=12,
                  fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG / "intervention_survival.png", dpi=140,
                bbox_inches="tight")
    plt.close()
    print("saved: intervention_survival.png")


def main():
    inter, entity, repair = load_results()
    fig_bars(inter, entity)
    fig_entity_heatmap()
    fig_survival(repair)


if __name__ == "__main__":
    main()
