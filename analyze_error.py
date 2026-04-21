import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel('计算结果.xlsx')

# 提取128Hz的计算误差
error_128hz = df['128Hz计算误差(ms)'].values

# 计算误差统计信息
mean_error = np.mean(error_128hz)
std_error = np.std(error_128hz)
max_error = np.max(np.abs(error_128hz))
min_error = np.min(np.abs(error_128hz))

# 计算误差小于0.1的比例
error_less_than_01 = np.sum(np.abs(error_128hz) < 0.1) / len(error_128hz) * 100

# 打印统计信息
print("128Hz计算误差统计:")
print(f"均值: {mean_error:.3f} ms")
print(f"标准差: {std_error:.3f} ms")
print(f"最大绝对误差: {max_error:.3f} ms")
print(f"最小绝对误差: {min_error:.3f} ms")
print(f"误差小于0.1ms的比例: {error_less_than_01:.2f}%")

# 查看前20个误差值
print("\n前20个误差值:")
print(error_128hz[:20])
