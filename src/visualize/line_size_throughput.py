import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("csv_path")


df['x_label'] = df['quantization_method'] + ': ' + df['file_size'].astype(str) + 'GB'
df = df[~df['quantization_method'].isin(["original", "original_ja", "F32", "F16"])]
df = df.sort_values('file_size')

sns.set_theme()

plt.figure(figsize=(12, 6))
plt.plot(df['x_label'], df['average_tps'], marker='o')

plt.xlabel('Quantization Method: File Size')
plt.ylabel('Average Throughput (TPS)')
plt.xticks(rotation=45, ha='right')
plt.grid(True)

plt.tight_layout()
plt.show()
