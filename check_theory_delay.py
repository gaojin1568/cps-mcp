import pandas as pd
import numpy as np

df_input = pd.read_excel('/workspace/模拟数据.xlsx')

Fs_ASSUMED = 1600.0
WINDOW_SIZE = int(Fs_ASSUMED * 0.5)
SLIDE_STEP_MS = 20.0
step = int(Fs_ASSUMED * (SLIDE_STEP_MS / 1000.0))

f_nom = 128.0
two_pi = 2 * np.pi
theory_signal = df_input[f'{int(f_nom)}Hz理论值'].values

print(f"分析{int(f_nom)}Hz理论信号的延迟")
print("=" * 80)

# 计算前几个窗口的理论延迟
for window_idx in range(1, 11):
    start_sample = (window_idx - 1) * step
    end_sample = start_sample + WINDOW_SIZE
    if end_sample > len(theory_signal):
        break
    
    # 提取窗口数据
    window_data = theory_signal[start_sample:end_sample]
    
    # 计算窗口的延迟
    # 使用最小二乘拟合
    t = np.arange(len(window_data)) / Fs_ASSUMED
    X = np.array([np.sin(2*np.pi*f_nom*t), np.cos(2*np.pi*f_nom*t)]).T
    a, b = np.linalg.lstsq(X, window_data, rcond=None)[0]
    
    # 计算相位和延迟
    phi = np.arctan2(b, a)
    delay = (phi / (2*np.pi*f_nom)) * 1000.0
    period = 1000.0 / f_nom
    delay = delay % period
    if delay < 0:
        delay += period
    
    # 计算理论延迟
    initial_delay = 4.0
    theoretical_delay = (initial_delay + (window_idx - 1) * SLIDE_STEP_MS) % period
    
    print(f"窗口 {window_idx}:")
    print(f"  计算延迟: {delay:.6f} ms")
    print(f"  理论延迟: {theoretical_delay:.6f} ms")
    print(f"  误差: {delay - theoretical_delay:.6f} ms")
    print()
