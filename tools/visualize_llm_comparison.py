import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


MODEL_NAMES = ["OPUS-MT", "GPT-4o", "Claude", "Gemini"]
MODEL_KEYS  = ["opus_mt", "gpt4o", "claude", "gemini"]
MODEL_COLORS = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]


def main():
    base = Path(__file__).parent.parent
    eval_dir = base / "evaluation"
    fig_dir = eval_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    data = json.loads((eval_dir / "llm_comparison.json").read_text(encoding="utf-8"))
    models = data["models"]

    # --- mean metrics
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (key, title, ymax) in zip(axes, [
        ("bleu", "BLEU Score", 70),
        ("bert_f1", "BERTScore-F1", 1),
        ("meteor", "METEOR", 1),
    ]):
        means = [models[k][key]["mean"] for k in MODEL_KEYS]
        stds  = [models[k][key]["std"]  for k in MODEL_KEYS]
        bars = ax.bar(MODEL_NAMES, means, color=MODEL_COLORS,
                      edgecolor='black', linewidth=1.2)
        ax.errorbar(MODEL_NAMES, means, yerr=stds, fmt='none',
                    color='black', capsize=5, capthick=2)
        ax.set_ylabel(title)
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontweight='bold')
        for bar, mean in zip(bars, means):
            ax.annotate(f'{mean:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "llm_comparison_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("llm_comparison_metrics.png")

    # --- improvement over OPUS-MT
    comparisons = data["comparisons"]
    llm_names = ["GPT-4o", "Claude", "Gemini"]
    llm_keys = ["opus_vs_gpt4o", "opus_vs_claude", "opus_vs_gemini"]
    metric_specs = [("bleu", "BLEU", "#1f77b4"),
                    ("bert_f1", "BERTScore-F1", "#ff7f0e"),
                    ("meteor", "METEOR", "#2ca02c")]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(llm_keys))
    width = 0.25
    for i, (key, label, color) in enumerate(metric_specs):
        vals = [comparisons[k][key]["improvement_percent"] for k in llm_keys]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, edgecolor='black')
        for bar, v in zip(bars, vals):
            ax.annotate(f'+{v:.0f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Improvement over OPUS-MT (%)')
    ax.set_title('Relative Improvement of LLM Translators vs OPUS-MT',
                 fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(llm_names)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 180)
    plt.tight_layout()
    plt.savefig(fig_dir / "llm_improvement_percent.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("llm_improvement_percent.png")

    # --- BLEU per segment
    dfs = {}
    for key, csv_name in [("OPUS-MT", "mt_metrics.csv"),
                          ("GPT-4o",  "mt_gpt4o_metrics.csv"),
                          ("Claude",  "mt_claude_metrics.csv"),
                          ("Gemini",  "mt_gemini_metrics.csv")]:
        dfs[key] = pd.read_csv(eval_dir / csv_name)[['segment_name', 'bleu']] \
                     .rename(columns={'bleu': key})

    merged = dfs["OPUS-MT"]
    for k in ("GPT-4o", "Claude", "Gemini"):
        merged = merged.merge(dfs[k], on='segment_name')

    merged['short'] = merged['segment_name'].apply(
        lambda s: s.replace('_seg', '\nseg').replace('_', ' ')[:20])

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(merged))
    width = 0.2
    for i, (m, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
        ax.bar(x + i * width, merged[m].values, width, label=m,
               color=color, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('BLEU Score')
    ax.set_title('BLEU Score by Segment: OPUS-MT vs LLM Translators',
                 fontweight='bold')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(merged['short'], fontsize=8, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 75)
    plt.tight_layout()
    plt.savefig(fig_dir / "llm_bleu_by_segment.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("llm_bleu_by_segment.png")


if __name__ == "__main__":
    main()
