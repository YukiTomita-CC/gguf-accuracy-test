import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("csv_path")


df['prefix'] = df['quantization_method'].str.split('-').str[0]
df['method'] = df['quantization_method'].str.split('-').str[1]

c4_data = df[df['prefix'] == 'C4'].sort_values('average_score', ascending=False)

method_order = c4_data['method'].tolist()

df['method'] = pd.Categorical(df['method'], categories=method_order, ordered=True)
df = df.sort_values('method')

df.to_csv('sorted_data.csv', index=False)

prefix_fullname = {
    'C4': 'c4_en_ja',
    'DL': 'dolly-15k',
    'TP': 'dolly-15k-prompt'
}

sns.set_theme()

plt.figure(figsize=(10, 6))

for prefix in ['C4', 'DL', 'TP']:
    data = df[df['prefix'] == prefix]
    plt.plot(data['method'], data['average_score'], marker='.', label=prefix_fullname[prefix])

plt.xlabel('Quantization Method')
plt.ylabel('Average Score')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()