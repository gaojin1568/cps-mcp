import pandas as pd
import numpy as np

df = pd.read_excel('/workspace/计算结果_优化版.xlsx')
df_input = pd.read_excel('/workspace/模拟数据.xlsx')

print("对比窗口计算延迟和理论延迟的前20个窗口:")
print("=" * 100)
freq = 128

calc_col = f"{freq}Hz窗口计算延迟(ms)"
theory_col = f"{freq}Hz窗口理论延迟(ms)"
error_col = f"{freq}Hz计算误差(ms)"

if calc_col in df.columns and theory_col in df.columns:
    print(f"{'窗口':<6}{'计算延迟':<12}{'理论延迟':<12}{'误差':<12}")
    print("-" * 50)
    
    for i in range(min(20, len(df))):
        calc = df[calc_col].iloc[i]
        theory = df[theory_col].iloc[i]
        error = df[error_col].iloc[i]
        print(f"{df['窗口'].iloc[i]:<6}{calc:<12.6f}{theory:<12.6f}{error:<12.6f}")

# 让我们用同样的方法重新计算理论延迟
print("\n" + "=" * 100)
print("用计算窗口计算延迟的同样方法计算理论值的延迟:")

Fs_ASSUMED = 1600.0
WINDOW_SIZE = int(Fs_ASSUMED * 0.5)
SLIDE_STEP_MS = 20.0
step = int(Fs_ASSUMED * (SLIDE_STEP_MS / 1000.0))
f_nom = 128.0
two_pi = 2 * np.pi

# 读取理论值
theory_signal = df_input[f'{int(f_nom)}Hz理论值'].values

# 预计算
raw_time_s = np.arange(WINDOW_SIZE, dtype=np.float32) / Fs_ASSUMED
t_center_window = (WINDOW_SIZE - 1) / (2.0 * Fs_ASSUMED)
time_s_rel = raw_time_s - t_center_window

print(f"\n使用同样的最小二乘法拟合方法计算理论值的延迟:")

# 计算几个窗口
for window_idx in range(1, 6):
    start_sample = (window_idx - 1) * step
    end_sample = start_sample + WINDOW_SIZE
    if end_sample > len(theory_signal):
        break
    
    S_window = theory_signal[start_sample:end_sample]
    
    # 拟合sin和cos
    t_rel = time_s_rel
    X = np.zeros((len(t_rel), 2), dtype=np.float32)
    wt = two_pi * f_nom * t_rel
    X[:, 0] = np.sin(wt)
    X[:, 1] = np.cos(wt)
    
    theta, residuals, rank, singular = np.linalg.lstsq(X, S_window, rcond=None)
    a = theta[0]
    b = theta[1]
    
    # 同样的延迟计算方法
    phi_center = -np.arctan2(b, a)
    t_center_absolute = (start_sample + (WINDOW_SIZE - 1) / 2.0) / Fs_ASSUMED
    phi_absolute = phi_center + two_pi * f_nom * t_center_absolute
    y_ms = (phi_absolute / (two_pi * f_nom)) * 1000.0
    period_ms = 1000.0 / f_nom
    delay_from_theory = y_ms % period_ms
    
    # 从结果文件中读取我们的计算延迟
    calc_delay = df[calc_col].iloc[window_idx - 1]
    theory_delay = df[theory_col].iloc[window_idx - 1]
    
    print(f"\n窗口 {window_idx}:")
    print(f"  用同样方法从理论值计算的延迟: {delay_from_theory:.6f} ms")
    print(f"  我们从实际信号计算的延迟: {calc_delay:.6f} ms")
    print(f"  直接从理论点算的延迟: {theory_delay:.6f} ms")
