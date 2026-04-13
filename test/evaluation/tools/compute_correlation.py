"""
Spearman and Pearson correlation between ASR and MT metrics.
"""

import json
import pandas as pd
from scipy import stats
from pathlib import Path

base_dir = Path(__file__).parent.parent
eval_dir = base_dir / "evaluation"

asr_df = pd.read_csv(eval_dir / "asr_metrics.csv")
mt_df = pd.read_csv(eval_dir / "mt_metrics.csv")
merged = pd.merge(asr_df, mt_df, on="segment_name")

print("=" * 60)
print("КОРЕЛЯЦІЙНИЙ АНАЛІЗ")
print("=" * 60)

print("\nДані:")
print(merged[["segment_name", "wer", "cer", "bleu", "bert_f1", "meteor"]].to_string(index=False))

print("\n" + "-" * 60)
print("Кореляція Спірмена")
print("-" * 60)

correlations = {}
for asr_metric in ["wer", "cer"]:
    correlations[asr_metric] = {}
    for mt_metric in ["bleu", "bert_f1", "meteor"]:
        rho, pvalue = stats.spearmanr(merged[asr_metric], merged[mt_metric])
        correlations[asr_metric][mt_metric] = {"rho": rho, "p": pvalue}
        sig = "*" if pvalue < 0.05 else ""
        print(f"{asr_metric} vs {mt_metric}: rho = {rho:.3f}, p = {pvalue:.3f} {sig}")

print("\n" + "-" * 60)
print("Кореляція Пірсона")
print("-" * 60)

for asr_metric in ["wer", "cer"]:
    for mt_metric in ["bleu", "bert_f1", "meteor"]:
        r, pvalue = stats.pearsonr(merged[asr_metric], merged[mt_metric])
        sig = "*" if pvalue < 0.05 else ""
        print(f"{asr_metric} vs {mt_metric}: r = {r:.3f}, p = {pvalue:.3f} {sig}")

print("\n" + "-" * 60)
print("Описова статистика")
print("-" * 60)
print(merged[["wer", "cer", "bleu", "bert_f1", "meteor"]].describe())

with open(eval_dir / "correlations.json", "w") as f:
    json.dump(correlations, f, indent=2)
print(f"\nЗбережено: {eval_dir / 'correlations.json'}")
