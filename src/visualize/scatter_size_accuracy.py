import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("csv_path")

size_reduction = df['size_reduction'].tolist()
accuracy_drop = df['accuracy_drop'].tolist()
methods = df['quantization_method'].tolist()

sns.set_theme()

fig, ax = plt.subplots(figsize=(10, 6))

color_map = {
    'Original': '#1f77b4',
    'Non-quantization': '#ff7f0e',
    'IMatrix-quantization': '#2ca02c',
    'Quantization': '#d62728'
}

colors = []
for method in methods:
    if method in ["original", "original_ja"]:
        colors.append(color_map['Original'])
    elif method in ["F32", "F16"]:
        colors.append(color_map['Non-quantization'])
    elif method.startswith("imatrix"):
        colors.append(color_map['IMatrix-quantization'])
    else:
        colors.append(color_map['Quantization'])

scatter = ax.scatter(accuracy_drop, size_reduction, s=100, c=colors)

for i, method in enumerate(methods):
    ax.annotate(method, (accuracy_drop[i], size_reduction[i]), xytext=(5, 5), 
                textcoords='offset points')

ax.set_xlabel('Accuracy Drop (%)')
ax.set_ylabel('Size Reduction (%)')
ax.grid(True)

ax.axhspan(min(size_reduction), max(size_reduction), 0, min(accuracy_drop), alpha=0.1, color='green')

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=group, 
                   markerfacecolor=color, markersize=10) for group, color in color_map.items()]
ax.legend(handles=legend_elements, loc='best')

ax.set_xlim(-5, 7.5)
ax.set_ylim(0, 55)

plt.tight_layout()
plt.show()
