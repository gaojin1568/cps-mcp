import pandas as pd
import numpy as np

df_input = pd.read_excel('/workspace/模拟数据.xlsx')

# 计算实际信号和理论信号的关系
print("分析实际信号和理论信号的关系")
print("=" * 80)

# 读取信号
actual_signal = df_input['实际信号'].values
freqs = [4, 8, 50, 128]
theory_signals = {}
for freq in freqs:
    theory_signals[freq] = df_input[f'{freq}Hz理论值'].values

# 计算前1000个点的相关性
print("信号相关性分析:")
for freq in freqs:
    correlation = np.corrcoef(actual_signal[:1000], theory_signals[freq][:1000])[0, 1]
    print(f"{freq}Hz: 相关性 = {correlation:.6f}")

# 分析128Hz的情况
print("\n128Hz信号分析:")
print("前10个点:")
print("  实际信号    理论信号    差值")
print("-" * 40)
for i in range(10):
    actual = actual_signal[i]
    theory = theory_signals[128][i]
    diff = actual - theory
    print(f"  {actual:10.6f} {theory:10.6f} {diff:10.6f}")
