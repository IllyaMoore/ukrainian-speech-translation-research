import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from scipy import stats

import sacrebleu
from bert_score import score as bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

for resource, name in [('corpora/wordnet', 'wordnet'),
                        ('corpora/wordnet', 'omw-1.4'),
                        ('tokenizers/punkt', 'punkt'),
                        ('tokenizers/punkt', 'punkt_tab')]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name)


@dataclass
class MTMetrics:
    segment_name: str
    bleu: float
    bert_precision: float
    bert_recall: float
    bert_f1: float
    meteor: float


def mt_metrics(reference, hypothesis, name):
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score
    P, R, F1 = bert_score([hypothesis], [reference],
                          lang="en", rescale_with_baseline=True, verbose=False)
    met = meteor_score([word_tokenize(reference.lower())],
                       word_tokenize(hypothesis.lower()))
    return MTMetrics(
        segment_name=name,
        bleu=round(bleu, 2),
        bert_precision=round(P[0].item(), 4),
        bert_recall=round(R[0].item(), 4),
        bert_f1=round(F1[0].item(), 4),
        meteor=round(met, 4),
    )


def load_texts(folder):
    return {f.stem: f.read_text(encoding="utf-8").strip()
            for f in folder.glob("*.txt")}


def evaluate_model(name, mt_dir, ref_en):
    texts = load_texts(mt_dir)
    common = sorted(set(texts) & set(ref_en))
    print(f"  {len(common)} спільних сегментів")
    rows = []
    for n in common:
        print(f"    {n}...", end=" ", flush=True)
        m = mt_metrics(ref_en[n], texts[n], n)
        rows.append(m)
        print(f"BLEU={m.bleu:.1f}, F1={m.bert_f1:.3f}")
    return rows


def stats_block(df):
    out = {}
    for col in ('bleu', 'bert_f1', 'meteor'):
        d = 2 if col == 'bleu' else 4
        out[col] = {
            "mean":   round(df[col].mean(),   d),
            "std":    round(df[col].std(),    d),
            "min":    round(df[col].min(),    d),
            "max":    round(df[col].max(),    d),
            "median": round(df[col].median(), d),
        }
    return out


def compare(opus_df, llm_df):
    out = {}
    for col in ('bleu', 'bert_f1', 'meteor'):
        a, b = opus_df[col].values, llm_df[col].values
        t_stat, t_p = stats.ttest_rel(a, b)
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = None, None
        diff = float(b.mean() - a.mean())
        out[col] = {
            "opus_mean": round(float(a.mean()), 4),
            "llm_mean":  round(float(b.mean()), 4),
            "difference": round(diff, 4),
            "improvement_percent":
                round(diff / a.mean() * 100, 2) if a.mean() else 0,
            "t_test": {
                "statistic": round(float(t_stat), 4),
                "p_value":   round(float(t_p), 4),
                "significant": bool(t_p < 0.05),
            },
            "wilcoxon": {
                "statistic": round(float(w_stat), 4) if w_stat is not None else None,
                "p_value":   round(float(w_p), 4) if w_p is not None else None,
                "significant": bool(w_p < 0.05) if w_p is not None else None,
            },
        }
    return out


def main():
    base = Path(__file__).parent.parent
    eval_dir = base / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    ref_en = load_texts(base / "Translations EN")
    print(f"Референсних EN: {len(ref_en)}")

    opus_csv = eval_dir / "mt_metrics.csv"
    if not opus_csv.exists():
        print("mt_metrics.csv не знайдено — спочатку запусти evaluate_metrics.py")
        return

    opus_df = pd.read_csv(opus_csv)
    print(f"OPUS-MT: {len(opus_df)} сегментів")

    all_stats = {"opus_mt": stats_block(opus_df)}
    comparisons = {}

    for model in ("gpt4o", "claude", "gemini"):
        mt_dir = base / "results" / f"mt_{model}"
        if not mt_dir.exists():
            print(f"\nпропуск {model}: {mt_dir} не існує")
            continue

        print(f"\n--- {model} ---")
        rows = evaluate_model(model, mt_dir, ref_en)
        if not rows:
            continue

        df = pd.DataFrame([asdict(m) for m in rows])
        df.to_csv(eval_dir / f"mt_{model}_metrics.csv", index=False)

        st = stats_block(df)
        all_stats[model] = st
        print(f"  BLEU mean={st['bleu']['mean']:.2f}, "
              f"F1 mean={st['bert_f1']['mean']:.4f}, "
              f"METEOR mean={st['meteor']['mean']:.4f}")

        merged = pd.merge(opus_df, df, on='segment_name', suffixes=('_opus', '_llm'))
        if len(merged):
            opus_part = merged[['bleu_opus', 'bert_f1_opus', 'meteor_opus']].rename(
                columns=lambda c: c.replace('_opus', ''))
            llm_part = merged[['bleu_llm', 'bert_f1_llm', 'meteor_llm']].rename(
                columns=lambda c: c.replace('_llm', ''))
            comparisons[f"opus_vs_{model}"] = compare(opus_part, llm_part)

    print(f"\n{'Model':<15}{'BLEU':>8}{'BERT-F1':>12}{'METEOR':>10}")
    for m, st in all_stats.items():
        print(f"{m:<15}{st['bleu']['mean']:>8.2f}"
              f"{st['bert_f1']['mean']:>12.4f}{st['meteor']['mean']:>10.4f}")

    summary = {"models": all_stats, "comparisons": comparisons,
               "segments_evaluated": len(opus_df)}
    (eval_dir / "llm_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if comparisons:
        print("\nСтат. значущість vs OPUS-MT:")
        for name, data in comparisons.items():
            print(f"  {name}:")
            for col, d in data.items():
                mark = " *" if d['t_test']['significant'] else ""
                print(f"    {col}: diff={d['difference']:+.2f} "
                      f"({d['improvement_percent']:+.1f}%), "
                      f"p={d['t_test']['p_value']:.4f}{mark}")


if __name__ == "__main__":
    main()
