import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


df = pd.read_csv("csv_path")
df = df.sort_values('average_score', ascending=False).reset_index(drop=True)

sns.set_theme()

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Quantization Method')
ax1.set_ylabel('Average Score', color=color)
ax1.plot(range(len(df)), df['average_score'], color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(df['quantization_method'], rotation=45, ha='right')

ax2 = ax1.twinx()
ax2.set_ylabel('Abnormal Responses', color='tab:blue')

width = 0.35

ax2.bar(np.arange(len(df)) - width/2, df['non_ja_responses'], width, label='Non-JA Responses', color='tab:blue', alpha=0.7)
ax2.bar(np.arange(len(df)) + width/2, df['infinite_repetitions'], width, label='Infinite Repetitions', color='tab:green', alpha=0.7)

ax2.set_ylim(0, 50)
ax2.tick_params(axis='y', labelcolor='tab:blue')

fig.tight_layout()

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=(0.75, 0.85))

plt.show()
