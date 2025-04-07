import matplotlib.pyplot as plt
import pandas as pd

# 读取 CSV 文件
file_path = "./test.csv"
data = pd.read_csv(file_path)

# 提取 x 轴数据
x = data.iloc[:, 0]  # 第一列作为 x 轴

# 选择指定的列
selected_columns = [
    'CM block_m=32 block_n=32 block_k=128',
    'CM block_m=64 block_n=32 block_k=128',
    'CM block_m=64 block_n=64 block_k=32'
]

# 自定义图例标签
legend_labels = [
    'block_m=32, n=32, k=128',
    'block_m=64, n=32, k=128',
    'block_m=64, n=64, k=32'
]

# 绘制折线图
plt.figure(figsize=(12, 8))
for col, label in zip(selected_columns, legend_labels):
    plt.plot(x, data[col], marker='o', label=label)

# 添加标题和标签
plt.title('Performance with different block size', fontsize=16)
plt.xlabel('Batch', fontsize=14)
plt.ylabel('TFLOPS', fontsize=14)

# 显示图例在绘图右侧
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

# 显示网格
plt.grid(True)

# 调整布局以适应图例
plt.tight_layout(rect=[0, 0, 0.85, 1])

# 保存图形
plt.savefig("test.png")