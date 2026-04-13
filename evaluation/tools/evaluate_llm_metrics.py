"""
Compute LLM translation metrics and compare with OPUS-MT baseline.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy import stats

import sacrebleu
from bert_score import score as bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


@dataclass
class MTMetrics:
    segment_name: str
    bleu: float
    bert_precision: float
    bert_recall: float
    bert_f1: float
    meteor: float


def compute_mt_metrics(reference: str, hypothesis: str, name: str) -> MTMetrics:
    bleu_result = sacrebleu.sentence_bleu(hypothesis, [reference])

    P, R, F1 = bert_score(
        [hypothesis], [reference],
        lang="en", rescale_with_baseline=True, verbose=False
    )

    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    meteor = meteor_score([ref_tokens], hyp_tokens)

    return MTMetrics(
        segment_name=name,
        bleu=round(bleu_result.score, 2),
        bert_precision=round(P[0].item(), 4),
        bert_recall=round(R[0].item(), 4),
        bert_f1=round(F1[0].item(), 4),
        meteor=round(meteor, 4)
    )


def load_texts(directory: Path) -> Dict[str, str]:
    texts = {}
    for txt_file in directory.glob("*.txt"):
        texts[txt_file.stem] = txt_file.read_text(encoding="utf-8").strip()
    return texts


def evaluate_model(model_name: str, mt_dir: Path, ref_en: Dict[str, str]) -> List[MTMetrics]:
    mt_texts = load_texts(mt_dir)
    common_segments = set(mt_texts.keys()) & set(ref_en.keys())
    print(f"  Спільних сегментів: {len(common_segments)}")

    metrics_list = []
    for name in sorted(common_segments):
        print(f"    {name}...", end=" ", flush=True)
        metrics = compute_mt_metrics(ref_en[name], mt_texts[name], name)
        metrics_list.append(metrics)
        print(f"BLEU={metrics.bleu:.1f}, BERTScore-F1={metrics.bert_f1:.3f}")

    return metrics_list


def compute_statistics(df: pd.DataFrame) -> dict:
    result = {}
    for col in ['bleu', 'bert_f1', 'meteor']:
        decimals = 2 if col == 'bleu' else 4
        result[col] = {
            "mean": round(df[col].mean(), decimals),
            "std": round(df[col].std(), decimals),
            "min": round(df[col].min(), decimals),
            "max": round(df[col].max(), decimals),
            "median": round(df[col].median(), decimals)
        }
    return result


def perform_statistical_tests(opus_df: pd.DataFrame, llm_df: pd.DataFrame, llm_name: str) -> dict:
    results = {}

    for metric in ['bleu', 'bert_f1', 'meteor']:
        opus_values = opus_df[metric].values
        llm_values = llm_df[metric].values

        t_stat, t_pvalue = stats.ttest_rel(opus_values, llm_values)

        try:
            w_stat, w_pvalue = stats.wilcoxon(opus_values, llm_values)
        except ValueError:
            w_stat, w_pvalue = None, None

        diff_mean = llm_values.mean() - opus_values.mean()

        results[metric] = {
            "opus_mean": round(float(opus_values.mean()), 4),
            "llm_mean": round(float(llm_values.mean()), 4),
            "difference": round(float(diff_mean), 4),
            "improvement_percent": round(float((diff_mean / opus_values.mean()) * 100), 2) if opus_values.mean() != 0 else 0,
            "t_test": {
                "statistic": round(float(t_stat), 4),
                "p_value": round(float(t_pvalue), 4),
                "significant": bool(t_pvalue < 0.05)
            },
            "wilcoxon": {
                "statistic": round(float(w_stat), 4) if w_stat else None,
                "p_value": round(float(w_pvalue), 4) if w_pvalue else None,
                "significant": bool(w_pvalue < 0.05) if w_pvalue else None
            }
        }

    return results


def main():
    base_dir = Path(__file__).parent.parent

    translations_en = base_dir / "Translations EN"
    results_dir = base_dir / "results"
    output_dir = base_dir / "evaluation"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("ОБЧИСЛЕННЯ МЕТРИК LLM-ПЕРЕКЛАДІВ")
    print("=" * 60)

    print("\nЗавантаження референсних EN текстів...")
    ref_en = load_texts(translations_en)
    print(f"  Знайдено: {len(ref_en)} файлів")

    opus_metrics_file = output_dir / "mt_metrics.csv"
    if opus_metrics_file.exists():
        opus_df = pd.read_csv(opus_metrics_file)
        print(f"\nЗавантажено OPUS-MT метрики: {len(opus_df)} сегментів")
    else:
        print("ПОМИЛКА: Не знайдено mt_metrics.csv для OPUS-MT")
        return

    all_results = {"opus_mt": compute_statistics(opus_df)}
    all_comparisons = {}

    for model_name in ["gpt4o", "claude", "gemini"]:
        mt_dir = results_dir / f"mt_{model_name}"
        if not mt_dir.exists():
            print(f"\nПапка {mt_dir} не існує, пропускаємо {model_name}")
            continue

        print(f"\n" + "-" * 60)
        print(f"ОЦІНКА {model_name.upper()}")
        print("-" * 60)

        metrics_list = evaluate_model(model_name, mt_dir, ref_en)
        if not metrics_list:
            continue

        df = pd.DataFrame([asdict(m) for m in metrics_list])
        csv_file = output_dir / f"mt_{model_name}_metrics.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n  Збережено: {csv_file}")

        model_stats = compute_statistics(df)
        all_results[model_name] = model_stats

        print(f"\n  BLEU:        mean={model_stats['bleu']['mean']:.2f}")
        print(f"  BERTScore-F1: mean={model_stats['bert_f1']['mean']:.4f}")
        print(f"  METEOR:       mean={model_stats['meteor']['mean']:.4f}")

        merged = pd.merge(opus_df, df, on='segment_name', suffixes=('_opus', '_llm'))
        if len(merged) > 0:
            opus_aligned = merged[['bleu_opus', 'bert_f1_opus', 'meteor_opus']].rename(
                columns={'bleu_opus': 'bleu', 'bert_f1_opus': 'bert_f1', 'meteor_opus': 'meteor'})
            llm_aligned = merged[['bleu_llm', 'bert_f1_llm', 'meteor_llm']].rename(
                columns={'bleu_llm': 'bleu', 'bert_f1_llm': 'bert_f1', 'meteor_llm': 'meteor'})
            all_comparisons[f"opus_vs_{model_name}"] = perform_statistical_tests(opus_aligned, llm_aligned, model_name)

    print("\n" + "=" * 60)
    print("ПОРІВНЯННЯ МОДЕЛЕЙ")
    print("=" * 60)

    print("\n{:<15} {:>10} {:>15} {:>10}".format("Model", "BLEU", "BERTScore-F1", "METEOR"))
    print("-" * 52)
    for model, st in all_results.items():
        print("{:<15} {:>10.2f} {:>15.4f} {:>10.4f}".format(
            model, st['bleu']['mean'], st['bert_f1']['mean'], st['meteor']['mean']
        ))

    summary = {
        "models": all_results,
        "comparisons": all_comparisons,
        "segments_evaluated": len(opus_df)
    }

    summary_file = output_dir / "llm_comparison.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nЗбережено: {summary_file}")

    if all_comparisons:
        print("\n" + "-" * 60)
        print("СТАТИСТИЧНА ЗНАЧУЩІСТЬ РІЗНИЦІ (vs OPUS-MT)")
        print("-" * 60)
        for comp_name, comp_data in all_comparisons.items():
            print(f"\n{comp_name}:")
            for metric, data in comp_data.items():
                sig = "***" if data['t_test']['significant'] else ""
                print(f"  {metric}: diff={data['difference']:+.2f} ({data['improvement_percent']:+.1f}%), p={data['t_test']['p_value']:.4f} {sig}")


if __name__ == "__main__":
    main()
