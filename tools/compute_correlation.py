import json
import pandas as pd
from scipy import stats
from pathlib import Path

eval_dir = Path(__file__).parent.parent / "evaluation"

asr = pd.read_csv(eval_dir / "asr_metrics.csv")
mt = pd.read_csv(eval_dir / "mt_metrics.csv")
df = pd.merge(asr, mt, on="segment_name")

print(df[["segment_name", "wer", "cer", "bleu", "bert_f1", "meteor"]].to_string(index=False))

asr_cols = ["wer", "cer"]
mt_cols = ["bleu", "bert_f1", "meteor"]

print("\nСпірмен:")
correlations = {}
for a in asr_cols:
    correlations[a] = {}
    for m in mt_cols:
        rho, p = stats.spearmanr(df[a], df[m])
        correlations[a][m] = {"rho": rho, "p": p}
        mark = " *" if p < 0.05 else ""
        print(f"  {a} vs {m}: rho={rho:+.3f}, p={p:.3f}{mark}")

print("\nПірсон:")
for a in asr_cols:
    for m in mt_cols:
        r, p = stats.pearsonr(df[a], df[m])
        mark = " *" if p < 0.05 else ""
        print(f"  {a} vs {m}: r={r:+.3f}, p={p:.3f}{mark}")

print()
print(df[asr_cols + mt_cols].describe())

with open(eval_dir / "correlations.json", "w") as f:
    json.dump(correlations, f, indent=2)
