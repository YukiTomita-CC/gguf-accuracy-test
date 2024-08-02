import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("csv_path")
df = df.sort_values('average_score', ascending=False)

average_score = df['average_score'].tolist()
max_deviation = df['max_deviation'].tolist()
min_deviation = df['min_deviation'].tolist()
methods = df['quantization_method'].tolist()

sns.set_theme()

fig, ax = plt.subplots(figsize=(12, 6))
x = range(1, len(average_score) + 1)

errors_minus = min_deviation
errors_plus = max_deviation

ax.errorbar(x, average_score, yerr=[errors_minus, errors_plus], fmt='.', capsize=5, capthick=2, ecolor='steelblue', markersize=10)

ax.axhline(y=3.03, color='green', linestyle='--', linewidth=2, label='Baseline (y=3.03)')

ax.set_xlim(0, len(average_score) + 1)
ax.set_ylim(min(average_score) - 0.2, max(average_score) + 0.2)
ax.set_xlabel('quantization_method')
ax.set_ylabel('average_score')
ax.grid(True)

ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right')

for i, score in enumerate(average_score):
    ax.annotate(f'{score:.2f}', (i + 1, score), xytext=(0, 10), 
                textcoords='offset points', ha='center', va='bottom')

plt.tight_layout()
plt.show()
