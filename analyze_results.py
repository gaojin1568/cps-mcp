import pandas as pd
import numpy as np

# 读取计算结果
df = pd.read_excel('计算结果.xlsx')

# 分析每个频率的误差
frequencies = [4, 8, 50, 128]

for freq in frequencies:
    error_col = f'{freq}Hz计算误差(ms)'
    if error_col in df.columns:
        errors = df[error_col].abs()
        mean_error = errors.mean()
        max_error = errors.max()
        min_error = errors.min()
        std_error = errors.std()
        
        print(f"\n{freq}Hz 误差分析:")
        print(f"平均误差: {mean_error:.4f} ms")
        print(f"最大误差: {max_error:.4f} ms")
        print(f"最小误差: {min_error:.4f} ms")
        print(f"误差标准差: {std_error:.4f} ms")
        print(f"误差小于0.1ms的比例: {(errors < 0.1).mean() * 100:.2f}%")
        
        # 检查是否所有误差都小于0.1ms
        if (errors < 0.1).all():
            print(f"✓ 所有{freq}Hz的误差都小于0.1ms")
        else:
            print(f"✗ 部分{freq}Hz的误差大于0.1ms")
            # 显示误差较大的窗口
            large_errors = df[errors >= 0.1]
            if len(large_errors) > 0:
                print(f"误差较大的窗口: {large_errors['窗口'].tolist()}")
                print(f"最大误差值: {max_error:.4f} ms")

# 分析理论延迟和计算延迟的关系
print("\n--- 理论延迟和计算延迟关系分析 ---")
for freq in frequencies:
    calc_col = f'{freq}Hz窗口计算延迟(ms)'
    theory_col = f'{freq}Hz窗口理论延迟(ms)'
    if calc_col in df.columns and theory_col in df.columns:
        # 计算两者的相关性
        correlation = df[calc_col].corr(df[theory_col])
        print(f"{freq}Hz 计算延迟与理论延迟的相关性: {correlation:.4f}")
        
        # 计算平均偏差
        mean_diff = (df[calc_col] - df[theory_col]).abs().mean()
        print(f"{freq}Hz 计算延迟与理论延迟的平均偏差: {mean_diff:.4f} ms")
