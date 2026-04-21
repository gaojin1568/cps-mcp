import pandas as pd
import numpy as np

df = pd.read_excel('/workspace/计算结果.xlsx')
print("优化后的误差统计:")
print("=" * 100)

for freq in [4, 8, 50, 128]:
    error_col = f"{freq}Hz计算误差(ms)"
    calc_col = f"{freq}Hz窗口计算延迟(ms)"
    theory_col = f"{freq}Hz窗口理论延迟(ms)"
    
    if error_col in df.columns:
        errors = df[error_col].dropna()
        print(f"\n{freq}Hz:")
        print(f"  样本数: {len(errors)}")
        print(f"  平均误差: {errors.mean():.9f} ms")
        print(f"  误差标准差: {errors.std():.9f} ms")
        print(f"  误差绝对值小于0.1ms的比例: {(errors.abs() < 0.1).mean() * 100:.2f}%")
        print(f"  最大误差绝对值: {errors.abs().max():.9f} ms")
        print(f"  误差绝对值小于0.01ms的比例: {(errors.abs() < 0.01).mean() * 100:.2f}%")
        
        if calc_col in df.columns and theory_col in df.columns:
            print(f"\n  前10个窗口的延迟对比:")
            print(f"  {'窗口':<6}{'计算延迟':<15}{'理论延迟':<15}{'误差':<15}")
            print(f"  {'-'*50}")
            for i in range(min(10, len(df))):
                calc = df[calc_col].iloc[i]
                theory = df[theory_col].iloc[i]
                error = df[error_col].iloc[i]
                print(f"  {df['窗口'].iloc[i]:<6}{calc:<15.6f}{theory:<15.6f}{error:<15.9f}")
