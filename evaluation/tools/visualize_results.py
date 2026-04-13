"""
Generate ASR and MT evaluation plots.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

base_dir = Path(__file__).parent.parent
eval_dir = base_dir / "evaluation"
fig_dir = eval_dir / "figures"
fig_dir.mkdir(exist_ok=True)

asr_df = pd.read_csv(eval_dir / "asr_metrics.csv")
mt_df = pd.read_csv(eval_dir / "mt_metrics.csv")
merged = pd.merge(asr_df, mt_df, on="segment_name")

SHORT_NAMES = {
    'В_Еміратах_сьогодні_seg001': 'Емірати-1',
    'В_Еміратах_сьогодні_seg002': 'Емірати-2',
    'Діалог_з_Америкою_seg001': 'Діалог-1',
    'Діалог_з_Америкою_seg002': 'Діалог-2',
    'Знаємо,_що_росіяни_seg001': 'Знаємо-1',
    'Знаємо,_що_росіяни_seg002': 'Знаємо-2',
    'Командна_ланка_seg001': 'Командна-1',
    'Наша_стратегія_seg001': 'Стратегія-1',
    'Наша_стратегія_seg002': 'Стратегія-2',
    'Отримуємо_хороші_seg001': 'Отримуємо-1',
    'Поведінка_seg001': 'Поведінка-1',
    'Поведінка_seg002': 'Поведінка-2',
}

def get_short_name(full_name):
    return SHORT_NAMES.get(full_name, full_name[:15])

asr_df['short_name'] = asr_df['segment_name'].apply(get_short_name)
mt_df['short_name'] = mt_df['segment_name'].apply(get_short_name)
merged['short_name'] = merged['segment_name'].apply(get_short_name)

print("=" * 60)
print("СТВОРЕННЯ ВІЗУАЛІЗАЦІЙ")
print("=" * 60)

# 1. ASR metrics by segment
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
colors = ['#e74c3c' if x > 0.1 else '#3498db' for x in asr_df['wer']]
ax1.barh(range(len(asr_df)), asr_df['wer'] * 100, color=colors)
ax1.set_yticks(range(len(asr_df)))
ax1.set_yticklabels(asr_df['short_name'], fontsize=10)
ax1.set_xlabel('WER (%)')
ax1.set_title('Word Error Rate (WER) за сегментами')
ax1.axvline(x=asr_df['wer'].mean() * 100, color='red', linestyle='--',
            label=f'Середнє: {asr_df["wer"].mean()*100:.1f}%')
ax1.legend()

ax2 = axes[1]
colors = ['#e74c3c' if x > 0.03 else '#2ecc71' for x in asr_df['cer']]
ax2.barh(range(len(asr_df)), asr_df['cer'] * 100, color=colors)
ax2.set_yticks(range(len(asr_df)))
ax2.set_yticklabels(asr_df['short_name'], fontsize=10)
ax2.set_xlabel('CER (%)')
ax2.set_title('Character Error Rate (CER) за сегментами')
ax2.axvline(x=asr_df['cer'].mean() * 100, color='red', linestyle='--',
            label=f'Середнє: {asr_df["cer"].mean()*100:.2f}%')
ax2.legend()

plt.tight_layout()
plt.savefig(fig_dir / 'asr_metrics_by_segment.png', dpi=150, bbox_inches='tight')
print(f"Збережено: {fig_dir / 'asr_metrics_by_segment.png'}")
plt.close()

# 2. MT metrics by segment
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for ax, col, title, thresholds in zip(
    axes,
    ['bleu', 'bert_f1', 'meteor'],
    ['BLEU за сегментами', 'BERTScore F1 за сегментами', 'METEOR за сегментами'],
    [(25, 20), (0.45, 0.35), (0.48, 0.4)]
):
    colors = ['#27ae60' if x > thresholds[0] else '#f39c12' if x > thresholds[1] else '#e74c3c' for x in mt_df[col]]
    ax.barh(range(len(mt_df)), mt_df[col], color=colors)
    ax.set_yticks(range(len(mt_df)))
    ax.set_yticklabels(mt_df['short_name'], fontsize=10)
    ax.set_xlabel(col.upper() if col != 'bert_f1' else 'BERTScore F1')
    ax.set_title(title)
    ax.axvline(x=mt_df[col].mean(), color='red', linestyle='--',
               label=f'Середнє: {mt_df[col].mean():.{"1f" if col == "bleu" else "3f"}')
    ax.legend()

plt.tight_layout()
plt.savefig(fig_dir / 'mt_metrics_by_segment.png', dpi=150, bbox_inches='tight')
print(f"Збережено: {fig_dir / 'mt_metrics_by_segment.png'}")
plt.close()

# 3. ASR vs MT correlation scatter
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, asr_metric in enumerate(['wer', 'cer']):
    for j, mt_metric in enumerate(['bleu', 'bert_f1', 'meteor']):
        ax = axes[i, j]
        x = merged[asr_metric] * 100
        y = merged[mt_metric]

        ax.scatter(x, y, s=80, alpha=0.7, c='#3498db', edgecolors='black')

        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        r, pval = pearsonr(x, y)
        sig = '*' if pval < 0.05 else ''

        ax.set_xlabel(f'{asr_metric.upper()} (%)')
        ax.set_ylabel(mt_metric.upper() if mt_metric != 'bert_f1' else 'BERTScore F1')
        ax.set_title(f'r = {r:.2f}{sig}')

plt.suptitle('Кореляція між метриками ASR та MT', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(fig_dir / 'correlation_scatter.png', dpi=150, bbox_inches='tight')
print(f"Збережено: {fig_dir / 'correlation_scatter.png'}")
plt.close()

# 4. Correlation heatmap
corr_data = merged[['wer', 'cer', 'bleu', 'bert_f1', 'meteor']].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0,
            fmt='.2f', square=True, linewidths=0.5,
            xticklabels=['WER', 'CER', 'BLEU', 'BERTScore', 'METEOR'],
            yticklabels=['WER', 'CER', 'BLEU', 'BERTScore', 'METEOR'])
ax.set_title('Кореляційна матриця метрик')
plt.tight_layout()
plt.savefig(fig_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
print(f"Збережено: {fig_dir / 'correlation_heatmap.png'}")
plt.close()

# 5. ASR error distribution
fig, ax = plt.subplots(figsize=(8, 6))

total_errors = asr_df['substitutions'].sum() + asr_df['deletions'].sum() + asr_df['insertions'].sum()
error_types = ['Заміни\n(Substitutions)', 'Видалення\n(Deletions)', 'Вставки\n(Insertions)']
error_counts = [asr_df['substitutions'].sum(), asr_df['deletions'].sum(), asr_df['insertions'].sum()]
error_pcts = [c/total_errors*100 for c in error_counts]

colors = ['#e74c3c', '#f39c12', '#3498db']
bars = ax.bar(error_types, error_counts, color=colors, edgecolor='black')

for bar, pct, count in zip(bars, error_pcts, error_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11)

ax.set_ylabel('Кількість помилок')
ax.set_title('Розподіл типів помилок ASR')
ax.set_ylim(0, max(error_counts) * 1.2)
plt.tight_layout()
plt.savefig(fig_dir / 'asr_error_distribution.png', dpi=150, bbox_inches='tight')
print(f"Збережено: {fig_dir / 'asr_error_distribution.png'}")
plt.close()

# 6. Boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
bp1 = ax1.boxplot([asr_df['wer'] * 100, asr_df['cer'] * 100],
                   labels=['WER (%)', 'CER (%)'], patch_artist=True)
bp1['boxes'][0].set_facecolor('#3498db')
bp1['boxes'][1].set_facecolor('#2ecc71')
ax1.set_title('Розподіл метрик ASR')
ax1.set_ylabel('Відсоток помилок')

ax2 = axes[1]
bp2 = ax2.boxplot([mt_df['bleu'], mt_df['bert_f1'] * 100, mt_df['meteor'] * 100],
                   labels=['BLEU', 'BERTScore×100', 'METEOR×100'], patch_artist=True)
bp2['boxes'][0].set_facecolor('#e74c3c')
bp2['boxes'][1].set_facecolor('#9b59b6')
bp2['boxes'][2].set_facecolor('#f39c12')
ax2.set_title('Розподіл метрик MT')
ax2.set_ylabel('Значення метрики')

plt.tight_layout()
plt.savefig(fig_dir / 'metrics_boxplot.png', dpi=150, bbox_inches='tight')
print(f"Збережено: {fig_dir / 'metrics_boxplot.png'}")
plt.close()

print("\n" + "=" * 60)
print(f"Всі графіки збережено в: {fig_dir}")
print("=" * 60)
