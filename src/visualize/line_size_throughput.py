import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text


df = pd.read_csv("csv_path")
df = df[~df['quantization_method'].isin(["original", "original_ja", "F32", "F16"])]
df = df.sort_values('file_size')

sns.set_theme()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(df['file_size'], df['average_tps'], marker='o')

texts = []
for i, row in df.iterrows():
    texts.append(ax.text(row['file_size'], row['average_tps'], row['quantization_method']))

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.5))

plt.xlabel('File Size (GB)')
plt.ylabel('Average TPS')

plt.grid(True)

plt.tight_layout()
plt.show()
