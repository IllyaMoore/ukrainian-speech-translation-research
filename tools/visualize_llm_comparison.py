"""
Generate OPUS-MT vs LLM comparison plots.
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def main():
    base_dir = Path(__file__).parent.parent
    eval_dir = base_dir / "evaluation"
    figures_dir = eval_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    with open(eval_dir / "llm_comparison.json", encoding="utf-8") as f:
        data = json.load(f)

    models = data["models"]

    # 1. Mean metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    model_names = ["OPUS-MT", "GPT-4o", "Claude", "Gemini"]
    model_keys = ["opus_mt", "gpt4o", "claude", "gemini"]
    colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]

    metrics = [
        ("bleu", "BLEU Score", 0, 70),
        ("bert_f1", "BERTScore-F1", 0, 1),
        ("meteor", "METEOR", 0, 1)
    ]

    for ax, (metric_key, metric_name, ymin, ymax) in zip(axes, metrics):
        means = [models[k][metric_key]["mean"] for k in model_keys]
        stds = [models[k][metric_key]["std"] for k in model_keys]

        bars = ax.bar(model_names, means, color=colors, edgecolor='black', linewidth=1.2)
        ax.errorbar(model_names, means, yerr=stds, fmt='none', color='black', capsize=5, capthick=2)

        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{metric_name}", fontsize=14, fontweight='bold')

        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{mean:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "llm_comparison_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Збережено: {figures_dir / 'llm_comparison_metrics.png'}")

    # 2. Improvement over OPUS-MT
    fig, ax = plt.subplots(figsize=(10, 6))

    comparisons = data["comparisons"]
    llm_names = ["GPT-4o", "Claude", "Gemini"]
    llm_keys = ["opus_vs_gpt4o", "opus_vs_claude", "opus_vs_gemini"]

    x = np.arange(3)
    width = 0.25

    metric_labels = ["BLEU", "BERTScore-F1", "METEOR"]
    metric_keys = ["bleu", "bert_f1", "meteor"]
    colors_metrics = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (metric_key, metric_label, color) in enumerate(zip(metric_keys, metric_labels, colors_metrics)):
        improvements = [comparisons[k][metric_key]["improvement_percent"] for k in llm_keys]
        bars = ax.bar(x + i*width, improvements, width, label=metric_label, color=color, edgecolor='black')

        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.annotate(f'+{imp:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Improvement over OPUS-MT (%)', fontsize=12)
    ax.set_title('Relative Improvement of LLM Translators vs OPUS-MT', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(llm_names, fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 180)

    plt.tight_layout()
    plt.savefig(figures_dir / "llm_improvement_percent.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Збережено: {figures_dir / 'llm_improvement_percent.png'}")

    # 3. BLEU by segment across models
    opus_df = pd.read_csv(eval_dir / "mt_metrics.csv")
    gpt4o_df = pd.read_csv(eval_dir / "mt_gpt4o_metrics.csv")
    claude_df = pd.read_csv(eval_dir / "mt_claude_metrics.csv")
    gemini_df = pd.read_csv(eval_dir / "mt_gemini_metrics.csv")

    merged = opus_df[['segment_name', 'bleu']].rename(columns={'bleu': 'OPUS-MT'})
    merged = merged.merge(gpt4o_df[['segment_name', 'bleu']].rename(columns={'bleu': 'GPT-4o'}), on='segment_name')
    merged = merged.merge(claude_df[['segment_name', 'bleu']].rename(columns={'bleu': 'Claude'}), on='segment_name')
    merged = merged.merge(gemini_df[['segment_name', 'bleu']].rename(columns={'bleu': 'Gemini'}), on='segment_name')

    merged['segment_short'] = merged['segment_name'].apply(lambda x: x.replace('_seg', '\nseg').replace('_', ' ')[:20])

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(merged))
    width = 0.2

    model_order = ['OPUS-MT', 'GPT-4o', 'Claude', 'Gemini']
    colors_models = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (model, color) in enumerate(zip(model_order, colors_models)):
        ax.bar(x + i*width, merged[model].values, width, label=model, color=color, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('BLEU Score by Segment: OPUS-MT vs LLM Translators', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(merged['segment_short'], fontsize=8, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 75)

    plt.tight_layout()
    plt.savefig(figures_dir / "llm_bleu_by_segment.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Збережено: {figures_dir / 'llm_bleu_by_segment.png'}")

    print("\nГотово! Всі графіки збережено.")


if __name__ == "__main__":
    main()
