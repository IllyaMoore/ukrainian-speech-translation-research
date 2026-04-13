import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'font.family': 'DejaVu Sans',
})
sns.set_style("whitegrid")

eval_dir = Path(__file__).parent.parent / "evaluation"
fig_dir = eval_dir / "figures"
fig_dir.mkdir(exist_ok=True)

asr_df = pd.read_csv(eval_dir / "asr_metrics.csv")
mt_df  = pd.read_csv(eval_dir / "mt_metrics.csv")
merged = pd.merge(asr_df, mt_df, on="segment_name")

SHORT = {
    'В_Еміратах_сьогодні_seg001': 'Емірати-1',
    'В_Еміратах_сьогодні_seg002': 'Емірати-2',
    'Діалог_з_Америкою_seg001':   'Діалог-1',
    'Діалог_з_Америкою_seg002':   'Діалог-2',
    'Знаємо,_що_росіяни_seg001':  'Знаємо-1',
    'Знаємо,_що_росіяни_seg002':  'Знаємо-2',
    'Командна_ланка_seg001':      'Командна-1',
    'Наша_стратегія_seg001':      'Стратегія-1',
    'Наша_стратегія_seg002':      'Стратегія-2',
    'Отримуємо_хороші_seg001':    'Отримуємо-1',
    'Поведінка_seg001':           'Поведінка-1',
    'Поведінка_seg002':           'Поведінка-2',
}

short = lambda n: SHORT.get(n, n[:15])
for d in (asr_df, mt_df, merged):
    d['short_name'] = d['segment_name'].apply(short)


def save(name):
    plt.tight_layout()
    plt.savefig(fig_dir / name, dpi=150, bbox_inches='tight')
    plt.close()
    print(name)


# WER + CER
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, col, thr, label, fmt in [
    (axes[0], 'wer', 0.10, 'WER (%)', '.1f'),
    (axes[1], 'cer', 0.03, 'CER (%)', '.2f'),
]:
    colors = ['#e74c3c' if x > thr else '#3498db' for x in asr_df[col]]
    ax.barh(range(len(asr_df)), asr_df[col] * 100, color=colors)
    ax.set_yticks(range(len(asr_df)))
    ax.set_yticklabels(asr_df['short_name'], fontsize=10)
    ax.set_xlabel(label)
    ax.set_title(f'{col.upper()} за сегментами')
    mean = asr_df[col].mean() * 100
    ax.axvline(mean, color='red', ls='--', label=f'Середнє: {mean:{fmt}}%')
    ax.legend()
save('asr_metrics_by_segment.png')


# MT bars
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
specs = [
    ('bleu',    'BLEU за сегментами',          (25, 20),       'BLEU',         '.1f'),
    ('bert_f1', 'BERTScore F1 за сегментами',  (0.45, 0.35),   'BERTScore F1', '.3f'),
    ('meteor',  'METEOR за сегментами',        (0.48, 0.4),    'METEOR',       '.3f'),
]
for ax, (col, title, (hi, lo), xlabel, fmt) in zip(axes, specs):
    colors = ['#27ae60' if x > hi else '#f39c12' if x > lo else '#e74c3c'
              for x in mt_df[col]]
    ax.barh(range(len(mt_df)), mt_df[col], color=colors)
    ax.set_yticks(range(len(mt_df)))
    ax.set_yticklabels(mt_df['short_name'], fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    mean = mt_df[col].mean()
    ax.axvline(mean, color='red', ls='--', label=f'Середнє: {mean:{fmt}}')
    ax.legend()
save('mt_metrics_by_segment.png')


# Correlation scatter
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, a in enumerate(['wer', 'cer']):
    for j, m in enumerate(['bleu', 'bert_f1', 'meteor']):
        ax = axes[i, j]
        x = merged[a] * 100
        y = merged[m]
        ax.scatter(x, y, s=80, alpha=0.7, c='#3498db', edgecolors='black')
        z = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), 'r--', alpha=0.8, lw=2)
        r, p = pearsonr(x, y)
        mark = '*' if p < 0.05 else ''
        ax.set_xlabel(f'{a.upper()} (%)')
        ax.set_ylabel('BERTScore F1' if m == 'bert_f1' else m.upper())
        ax.set_title(f'r = {r:.2f}{mark}')
plt.suptitle('Кореляція між метриками ASR та MT', fontsize=16, y=1.02)
save('correlation_scatter.png')


# Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['WER', 'CER', 'BLEU', 'BERTScore', 'METEOR']
sns.heatmap(merged[['wer', 'cer', 'bleu', 'bert_f1', 'meteor']].corr(),
            annot=True, cmap='RdBu_r', center=0, fmt='.2f',
            square=True, linewidths=0.5,
            xticklabels=labels, yticklabels=labels)
ax.set_title('Кореляційна матриця метрик')
save('correlation_heatmap.png')


# ASR error distribution
fig, ax = plt.subplots(figsize=(8, 6))
counts = [asr_df['substitutions'].sum(),
          asr_df['deletions'].sum(),
          asr_df['insertions'].sum()]
total = sum(counts)
labels = ['Заміни\n(Substitutions)', 'Видалення\n(Deletions)', 'Вставки\n(Insertions)']
bars = ax.bar(labels, counts, color=['#e74c3c', '#f39c12', '#3498db'], edgecolor='black')
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{c}\n({c/total*100:.1f}%)', ha='center', va='bottom', fontsize=11)
ax.set_ylabel('Кількість помилок')
ax.set_title('Розподіл типів помилок ASR')
ax.set_ylim(0, max(counts) * 1.2)
save('asr_error_distribution.png')


# Boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bp = axes[0].boxplot([asr_df['wer'] * 100, asr_df['cer'] * 100],
                     labels=['WER (%)', 'CER (%)'], patch_artist=True)
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#2ecc71')
axes[0].set_title('Розподіл метрик ASR')
axes[0].set_ylabel('Відсоток помилок')

bp = axes[1].boxplot([mt_df['bleu'], mt_df['bert_f1'] * 100, mt_df['meteor'] * 100],
                     labels=['BLEU', 'BERTScore×100', 'METEOR×100'], patch_artist=True)
for box, color in zip(bp['boxes'], ['#e74c3c', '#9b59b6', '#f39c12']):
    box.set_facecolor(color)
axes[1].set_title('Розподіл метрик MT')
axes[1].set_ylabel('Значення метрики')
save('metrics_boxplot.png')

print(f"\n-> {fig_dir}")
