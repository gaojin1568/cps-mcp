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

# 计算理论延迟，用最小二乘法拟合每个窗口
raw_time_s = np.arange(WINDOW_SIZE, dtype=np.float32) / Fs_ASSUMED
t_center_window = (WINDOW_SIZE - 1) / (2.0 * Fs_ASSUMED)
time_s_rel = raw_time_s - t_center_window

print(f"{f_nom}Hz，用最小二乘法拟合每个窗口的延迟Y:")
print("=" * 80)
print(f"{'窗口':<6}{'起始样本':<10}{'窗口中心样本':<16}{'计算延迟Y(ms)':<20}")
print("-" * 70)

for window_idx in range(1, 11):
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
    delay = y_ms % period_ms
    
    center_sample = start_sample + (WINDOW_SIZE - 1) // 2
    
    print(f"{window_idx:<6}{start_sample:<10}{center_sample:<16}{delay:<20.9f}")
