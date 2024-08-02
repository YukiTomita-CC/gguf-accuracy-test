import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("csv_path")

size_reduction = df['size_reduction'].tolist()
accuracy_drop = df['accuracy_drop'].tolist()
methods = df['quantization_method'].tolist()

sns.set_theme()

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(accuracy_drop, size_reduction, s=100)

for i, method in enumerate(methods):
    ax.annotate(method, (accuracy_drop[i], size_reduction[i]), xytext=(5, 5), 
                textcoords='offset points')

ax.set_xlabel('Accuracy Drop (%)')
ax.set_ylabel('Size Reduction (%)')
ax.set_title('Trade-off between Model Size and Accuracy')
ax.grid(True)

ax.axhspan(min(size_reduction), max(size_reduction), 0, min(accuracy_drop), alpha=0.1, color='green')

# ax.set_xlim(-5, 7.5)
# ax.set_ylim(0, 55)

plt.tight_layout()
plt.show()
