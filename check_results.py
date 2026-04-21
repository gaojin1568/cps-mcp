import pandas as pd
import numpy as np

# 读取计算结果
df_result = pd.read_excel('/workspace/计算结果.xlsx')
df_input = pd.read_excel('/workspace/模拟数据.xlsx')

print("计算结果列名:")
print(df_result.columns.tolist())

print("\n前10行计算结果:")
print(df_result.head(10))

print("\n128Hz计算误差统计:")
if '128Hz计算误差(ms)' in df_result.columns:
    print(df_result['128Hz计算误差(ms)'].describe())
    print(f"误差绝对值小于0.1的比例: {(df_result['128Hz计算误差(ms)'].abs() < 0.1).mean() * 100:.2f}%")

print("\n查看输入数据中理论值的相位关系:")
for freq in [4, 8, 50, 128]:
    col_name = f'{freq}Hz理论值'
    if col_name in df_input.columns:
        first_val = df_input[col_name].iloc[0]
        second_val = df_input[col_name].iloc[1]
        print(f"\n{freq}Hz:")
        print(f"  第0点: {first_val:.6f}")
        print(f"  第1点: {second_val:.6f}")
