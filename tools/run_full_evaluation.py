"""
Уніфікований скрипт повної оцінки якості перекладу.

Виконує всі етапи в одному процесі:
1. ASR метрики (WER, CER)
2. MT метрики (BLEU, BERTScore, METEOR) для OPUS-MT + усіх LLM-систем
3. Статистичні тести (paired t-test, Wilcoxon) — порівняння з OPUS-MT
4. Кореляційний аналіз ASR↔MT (Pearson + Spearman)
5. Візуалізація (bar-графіки, scatter, heatmap, boxplot)
6. Зведений JSON-звіт + готові LaTeX-таблиці

Використання:
    python tools/run_full_evaluation.py                      # лише OPUS (default)
    python tools/run_full_evaluation.py --systems claude     # лише Claude
    python tools/run_full_evaluation.py --systems opus haiku # дві системи
    python tools/run_full_evaluation.py --all                # всі доступні
    python tools/run_full_evaluation.py --skip asr viz       # без ASR та графіків
    python tools/run_full_evaluation.py --quick              # без BERTScore
"""

from __future__ import annotations

import argparse
import json
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_metrics import (  # noqa: E402
    compute_asr_metrics,
    compute_mt_metrics,
    load_reference_texts,
    load_pipeline_results,
)
from evaluate_llm_metrics import load_texts  # noqa: E402


KNOWN_LLM_SYSTEMS = ["claude", "gemini", "gpt4o", "haiku"]
ALL_SYSTEMS = ["opus"] + KNOWN_LLM_SYSTEMS


def stage_asr(
    ref_ua: Dict[str, str],
    pipeline_results: Dict[str, dict],
    output_dir: Path,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(" [1] ASR МЕТРИКИ (WER, CER)")
    print("=" * 70)

    common = sorted(set(ref_ua.keys()) & set(pipeline_results.keys()))
    metrics_list = []
    for name in common:
        ref = ref_ua[name]
        hyp = pipeline_results[name].get("full_text_uk", "")
        if not hyp:
            print(f"  [skip] {name}: немає ASR-тексту")
            continue
        m = compute_asr_metrics(ref, hyp, name)
        metrics_list.append(m)
        print(f"  {name}: WER={m.wer:.2%}, CER={m.cer:.2%}")

    df = pd.DataFrame([asdict(m) for m in metrics_list])
    df.to_csv(output_dir / "asr_metrics.csv", index=False)
    print(f"\n  → {output_dir / 'asr_metrics.csv'} ({len(df)} сегментів)")
    return df


def stage_mt(
    system: str,
    ref_en: Dict[str, str],
    pipeline_results: Dict[str, dict],
    results_dir: Path,
    output_dir: Path,
    quick: bool = False,
) -> Optional[pd.DataFrame]:
    """opus читається з results/*.json; інші — з results/mt_{system}/*.txt."""
    print("\n" + "-" * 70)
    print(f" MT-система: {system.upper()}")
    print("-" * 70)

    if system == "opus":
        hyp_texts = {
            name: r.get("full_text_en", "")
            for name, r in pipeline_results.items()
            if r.get("full_text_en")
        }
        csv_name = "mt_metrics.csv"
    else:
        mt_dir = results_dir / f"mt_{system}"
        if not mt_dir.exists():
            print(f"  [skip] папки {mt_dir} не існує")
            return None
        hyp_texts = load_texts(mt_dir)
        csv_name = f"mt_{system}_metrics.csv"

    common = sorted(set(hyp_texts.keys()) & set(ref_en.keys()))
    if not common:
        print(f"  [skip] немає спільних сегментів між {system} та референсами EN")
        return None

    metrics_list = []
    for name in common:
        print(f"  {name}...", end=" ", flush=True)
        if quick:
            import sacrebleu
            from nltk.translate.meteor_score import meteor_score
            from nltk.tokenize import word_tokenize
            bleu = sacrebleu.sentence_bleu(hyp_texts[name], [ref_en[name]]).score
            meteor = meteor_score(
                [word_tokenize(ref_en[name].lower())],
                word_tokenize(hyp_texts[name].lower()),
            )
            from evaluate_metrics import MTMetrics
            m = MTMetrics(
                segment_name=name,
                bleu=round(bleu, 2),
                bert_precision=0.0,
                bert_recall=0.0,
                bert_f1=0.0,
                meteor=round(meteor, 4),
            )
        else:
            m = compute_mt_metrics(ref_en[name], hyp_texts[name], name)
        metrics_list.append(m)
        print(f"BLEU={m.bleu:.1f} BERT-F1={m.bert_f1:.3f} METEOR={m.meteor:.3f}")

    df = pd.DataFrame([asdict(m) for m in metrics_list])
    df.to_csv(output_dir / csv_name, index=False)
    print(f"\n  → {output_dir / csv_name} ({len(df)} сегментів)")
    return df


def stage_system_comparison(
    mt_dfs: Dict[str, pd.DataFrame], baseline: str = "opus"
) -> dict:
    print("\n" + "=" * 70)
    print(f" [3] СТАТИСТИЧНЕ ПОРІВНЯННЯ MT-СИСТЕМ (baseline = {baseline})")
    print("=" * 70)

    if baseline not in mt_dfs:
        print(f"  [skip] baseline '{baseline}' недоступний")
        return {}

    base_df = mt_dfs[baseline]
    comparisons = {}

    for sys_name, sys_df in mt_dfs.items():
        if sys_name == baseline:
            continue
        merged = pd.merge(
            base_df, sys_df, on="segment_name", suffixes=("_base", "_sys")
        )
        if len(merged) < 3:
            print(f"  {sys_name}: мало спільних сегментів ({len(merged)}) для тестів")
            continue

        sys_results = {}
        for metric in ["bleu", "bert_f1", "meteor"]:
            base_vals = merged[f"{metric}_base"].values
            sys_vals = merged[f"{metric}_sys"].values
            if base_vals.std() == 0 and sys_vals.std() == 0:
                continue
            diff = sys_vals.mean() - base_vals.mean()

            try:
                t_stat, t_p = stats.ttest_rel(base_vals, sys_vals)
            except Exception:
                t_stat, t_p = float("nan"), float("nan")
            try:
                w_stat, w_p = stats.wilcoxon(base_vals, sys_vals)
            except Exception:
                w_stat, w_p = float("nan"), float("nan")

            sys_results[metric] = {
                "base_mean": round(float(base_vals.mean()), 4),
                "sys_mean": round(float(sys_vals.mean()), 4),
                "diff": round(float(diff), 4),
                "t_stat": round(float(t_stat), 4),
                "t_pvalue": round(float(t_p), 4),
                "wilcoxon_stat": round(float(w_stat), 4),
                "wilcoxon_pvalue": round(float(w_p), 4),
                "significant_0.05": bool(t_p < 0.05),
            }
        comparisons[f"{baseline}_vs_{sys_name}"] = sys_results

        print(f"\n  {baseline} vs {sys_name}:")
        for metric, data in sys_results.items():
            sig = "***" if data["significant_0.05"] else ""
            print(
                f"    {metric:8s}: {data['base_mean']:.3f} → {data['sys_mean']:.3f} "
                f"(Δ={data['diff']:+.3f}, p={data['t_pvalue']:.4f}) {sig}"
            )

    return comparisons


def stage_correlations(
    asr_df: pd.DataFrame, mt_df: pd.DataFrame, output_dir: Path
) -> dict:
    print("\n" + "=" * 70)
    print(" [4] КОРЕЛЯЦІЙНИЙ АНАЛІЗ (ASR ↔ MT)")
    print("=" * 70)

    merged = pd.merge(asr_df, mt_df, on="segment_name")
    if len(merged) < 3:
        print("  [skip] мало спільних сегментів для кореляцій")
        return {}

    correlations = {"pearson": {}, "spearman": {}}
    for asr_m in ["wer", "cer"]:
        correlations["pearson"][asr_m] = {}
        correlations["spearman"][asr_m] = {}
        for mt_m in ["bleu", "bert_f1", "meteor"]:
            if merged[mt_m].std() == 0:
                continue
            r, rp = stats.pearsonr(merged[asr_m], merged[mt_m])
            rho, sp = stats.spearmanr(merged[asr_m], merged[mt_m])
            correlations["pearson"][asr_m][mt_m] = {
                "r": round(float(r), 3),
                "p": round(float(rp), 4),
                "significant_0.05": bool(rp < 0.05),
            }
            correlations["spearman"][asr_m][mt_m] = {
                "rho": round(float(rho), 3),
                "p": round(float(sp), 4),
                "significant_0.05": bool(sp < 0.05),
            }
            sig = "*" if rp < 0.05 else ""
            print(f"  {asr_m} ↔ {mt_m}: r={r:+.3f} (p={rp:.3f}) {sig} | "
                  f"rho={rho:+.3f} (p={sp:.3f})")

    with open(output_dir / "correlations.json", "w", encoding="utf-8") as f:
        json.dump(correlations, f, ensure_ascii=False, indent=2)
    print(f"\n  → {output_dir / 'correlations.json'}")
    return correlations


def stage_visualizations(output_dir: Path) -> None:
    print("\n" + "=" * 70)
    print(" [5] ВІЗУАЛІЗАЦІЯ")
    print("=" * 70)
    import runpy
    viz_script = Path(__file__).parent / "visualize_results.py"
    try:
        runpy.run_path(str(viz_script), run_name="__main__")
    except SystemExit:
        pass


def _stats(series: pd.Series, digits: int = 2) -> str:
    return f"{series.mean():.{digits}f} \\pm {series.std():.{digits}f}"


def generate_latex_tables(
    asr_df: Optional[pd.DataFrame],
    mt_dfs: Dict[str, pd.DataFrame],
    comparisons: dict,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 70)
    print(" [6] ГЕНЕРАЦІЯ LaTeX-ТАБЛИЦЬ")
    print("=" * 70)

    lines: List[str] = [
        "% Згенеровано run_full_evaluation.py — НЕ редагувати вручну.",
        "% Для використання: \\input{evaluation/latex_tables.tex}",
        "",
    ]

    if asr_df is not None and len(asr_df) > 0:
        lines += [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Метрики ASR (Whisper medium)}",
            "\\label{tab:asr_summary}",
            "\\begin{tabular}{|l|c|c|c|c|}",
            "\\hline",
            "\\textbf{Метрика} & \\textbf{Середнє} & \\textbf{СКВ} & \\textbf{Мін} & \\textbf{Макс} \\\\",
            "\\hline",
            f"WER & {asr_df['wer'].mean():.4f} & {asr_df['wer'].std():.4f} & "
            f"{asr_df['wer'].min():.4f} & {asr_df['wer'].max():.4f} \\\\",
            f"CER & {asr_df['cer'].mean():.4f} & {asr_df['cer'].std():.4f} & "
            f"{asr_df['cer'].min():.4f} & {asr_df['cer'].max():.4f} \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]

    if mt_dfs:
        lines += [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Порівняння MT-систем за метриками якості (mean $\\pm$ std)}",
            "\\label{tab:mt_systems}",
            "\\begin{tabular}{|l|c|c|c|}",
            "\\hline",
            "\\textbf{Система} & \\textbf{BLEU} & \\textbf{BERTScore F1} & \\textbf{METEOR} \\\\",
            "\\hline",
        ]
        for sys_name in ALL_SYSTEMS:
            if sys_name not in mt_dfs:
                continue
            df = mt_dfs[sys_name]
            lines.append(
                f"{sys_name.upper()} & ${_stats(df['bleu'], 2)}$ & "
                f"${_stats(df['bert_f1'], 4)}$ & ${_stats(df['meteor'], 4)}$ \\\\"
            )
        lines += [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]

    if comparisons:
        lines += [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Статистична значущість різниці MT-систем відносно OPUS-MT (paired t-test)}",
            "\\label{tab:mt_significance}",
            "\\begin{tabular}{|l|c|c|c|}",
            "\\hline",
            "\\textbf{Порівняння} & \\textbf{$\\Delta$BLEU} & "
            "\\textbf{$\\Delta$BERT-F1} & \\textbf{$\\Delta$METEOR} \\\\",
            "\\hline",
        ]
        for comp_name, data in comparisons.items():
            row = [comp_name.replace("_", r"\_")]
            for metric in ["bleu", "bert_f1", "meteor"]:
                d = data.get(metric, {})
                if not d:
                    row.append("---")
                    continue
                mark = "$^{*}$" if d["significant_0.05"] else ""
                row.append(f"{d['diff']:+.3f} (p={d['t_pvalue']:.3f}){mark}")
            lines.append(" & ".join(row) + " \\\\")
        lines += [
            "\\hline",
            "\\end{tabular}",
            "\\\\[2pt]",
            "\\footnotesize $^{*}$ --- $p < 0{,}05$",
            "\\end{table}",
            "",
        ]

    tex_path = output_dir / "latex_tables.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {tex_path}  ({len(lines)} рядків)")


def _df_stats(df: pd.DataFrame, cols: List[str], digits: int = 4) -> dict:
    out = {}
    for c in cols:
        s = df[c]
        out[c] = {
            "mean": round(float(s.mean()), digits),
            "std": round(float(s.std()), digits),
            "min": round(float(s.min()), digits),
            "max": round(float(s.max()), digits),
            "median": round(float(s.median()), digits),
        }
    return out


def build_final_report(
    asr_df: Optional[pd.DataFrame],
    mt_dfs: Dict[str, pd.DataFrame],
    comparisons: dict,
    correlations: dict,
    duration_sec: float,
) -> dict:
    report = {
        "meta": {
            "duration_sec": round(duration_sec, 1),
            "systems_evaluated": list(mt_dfs.keys()),
        },
        "asr": None,
        "mt": {},
        "comparisons": comparisons,
        "correlations": correlations,
    }
    if asr_df is not None and len(asr_df) > 0:
        report["asr"] = {
            "segments": len(asr_df),
            **_df_stats(asr_df, ["wer", "cer"]),
            "total_substitutions": int(asr_df["substitutions"].sum()),
            "total_deletions": int(asr_df["deletions"].sum()),
            "total_insertions": int(asr_df["insertions"].sum()),
        }
    for sys_name, df in mt_dfs.items():
        report["mt"][sys_name] = {
            "segments": len(df),
            **_df_stats(df, ["bleu", "bert_f1", "meteor"]),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Повна оцінка якості ASR+MT pipeline"
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["asr", "mt", "compare", "corr", "viz"],
        help="Пропустити етапи",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["opus"],
        help=f"MT-системи. Доступні: {' '.join(ALL_SYSTEMS)}. Для всіх: --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Ярлик для --systems {' '.join(ALL_SYSTEMS)}",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Без BERTScore (швидко)",
    )
    args = parser.parse_args()
    if args.all:
        args.systems = ALL_SYSTEMS[:]

    unknown = [s for s in args.systems if s not in ALL_SYSTEMS]
    if unknown:
        print(
            f"ПОМИЛКА: невідомі системи: {unknown}. "
            f"Доступні: {ALL_SYSTEMS}",
            file=sys.stderr,
        )
        return 2

    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results"
    output_dir = base_dir / "evaluation"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(" ПОВНА ОЦІНКА ЯКОСТІ UA→EN PIPELINE")
    print("=" * 70)
    print(f" Skip: {args.skip or '—'}")
    print(f" Systems: {' '.join(args.systems)}")
    print(f" Quick mode: {args.quick}")

    t0 = time.perf_counter()

    print("\n Завантаження даних...")
    ref_ua = load_reference_texts(base_dir / "Translations UA")
    ref_en = load_reference_texts(base_dir / "Translations EN")
    pipeline_results = load_pipeline_results(results_dir)
    print(f"  refs UA: {len(ref_ua)}, refs EN: {len(ref_en)}, "
          f"pipeline: {len(pipeline_results)}")

    asr_df = None
    if "asr" not in args.skip:
        asr_df = stage_asr(ref_ua, pipeline_results, output_dir)

    mt_dfs: Dict[str, pd.DataFrame] = {}
    if "mt" not in args.skip:
        print("\n" + "=" * 70)
        print(" [2] MT МЕТРИКИ")
        print("=" * 70)
        for sys_name in args.systems:
            df = stage_mt(
                sys_name, ref_en, pipeline_results,
                results_dir, output_dir, quick=args.quick
            )
            if df is not None and len(df) > 0:
                mt_dfs[sys_name] = df

    comparisons = {}
    if "compare" not in args.skip and len(mt_dfs) >= 2:
        comparisons = stage_system_comparison(mt_dfs, baseline="opus")

    correlations = {}
    if "corr" not in args.skip and asr_df is not None and "opus" in mt_dfs:
        correlations = stage_correlations(asr_df, mt_dfs["opus"], output_dir)

    if "viz" not in args.skip:
        stage_visualizations(output_dir)

    generate_latex_tables(asr_df, mt_dfs, comparisons, output_dir)

    duration = time.perf_counter() - t0
    report = build_final_report(asr_df, mt_dfs, comparisons, correlations, duration)
    report_path = output_dir / "full_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f" ГОТОВО за {duration:.1f}s")
    print("=" * 70)
    print(f"  Повний звіт:     {report_path}")
    print(f"  LaTeX-таблиці:   {output_dir / 'latex_tables.tex'}")
    print(f"  ASR CSV:         {output_dir / 'asr_metrics.csv'}")
    print(f"  MT CSV:          {output_dir}/mt_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
