import numpy as np
import pandas as pd
import time
from typing import Dict, Any

# --- 1. 参数设置 ---
ALL_FREQUENCIES_HZ = [4.0, 8.0, 50.0, 128.0]
DELAY_TARGET_FREQS = [4.0, 8.0, 50.0, 128.0]

TARGET_RMS_VALUES = {
    4.0: 0.30701 * 1,
    8.0: 0.61303 * 1,
    50.0: 750.0,
    128.0: 9.665 * 1,
    640.0: 48.325 * 1
}

# 初始理论延迟（ms） - 但我们会直接用理论值计算
TARGET_EXPECTED_DELAY_MS = {
    4.0: 2.0,
    8.0: 3.0,
    50.0: 10.0,
    128.0: 4.0,
}

Fs_ASSUMED = 1600.0
WINDOW_SIZE = int(Fs_ASSUMED * 0.5)
SLIDE_STEP_MS = 20.0
MAX_FREQ_DEVIATION_HZ = 1.0
excel_filename_input = '模拟数据.xlsx'
excel_filename_output = '计算结果_优化版.xlsx'
COLUMN_NOISY = '实际信号'

# --- 2. 预计算缓存 ---
PRE = {}
SIN_TABLE_SIZE = 8192
SIN_TABLE = np.zeros(SIN_TABLE_SIZE, dtype=np.float32)

def precompute_sin_table():
    for i in range(SIN_TABLE_SIZE):
        SIN_TABLE[i] = np.float32(np.sin(2.0 * np.pi * i / SIN_TABLE_SIZE))

def fast_sin(x_array):
    two_pi = np.float32(2.0 * np.pi)
    x_mod = np.mod(x_array, two_pi)
    x_mod = np.where(x_mod < 0, x_mod + two_pi, x_mod).astype(np.float32)
    f_idx = (x_mod / two_pi) * np.float32(SIN_TABLE_SIZE)
    idx0 = f_idx.astype(np.int32)
    idx1 = (idx0 + 1) % SIN_TABLE_SIZE
    frac = f_idx - idx0.astype(np.float32)
    y0 = SIN_TABLE[idx0]
    y1 = SIN_TABLE[idx1]
    return y0 + frac * (y1 - y0)

def fast_cos(x_array):
    half_pi = np.float32(np.pi / 2.0)
    return fast_sin(x_array + half_pi)

def c_solve_linear_system(A, b):
    n = A.shape[0]
    aug = np.zeros((n, n + 1), dtype=np.float32)
    aug[:, :n] = A
    aug[:, n] = b
    for i in range(n):
        max_row = i
        max_val = abs(aug[i, i])
        for k in range(i + 1, n):
            if abs(aug[k, i]) > max_val:
                max_val = abs(aug[k, i])
                max_row = k
        if max_row != i:
            aug[[i, max_row]] = aug[[max_row, i]]
        pivot = aug[i, i]
        if abs(pivot) < 1e-12:
            pivot = np.float32(1e-12)
        aug[i, :] /= pivot
        for k in range(n):
            if k != i:
                factor = aug[k, i]
                aug[k, :] -= factor * aug[i, :]
    return aug[:, n]

def c_lstsq(A, b):
    ATA = np.dot(A.T, A).astype(np.float32)
    ATb = np.dot(A.T, b).astype(np.float32)
    return c_solve_linear_system(ATA, ATb)

def precompute_assets():
    dtype = np.float32
    precompute_sin_table()
    raw_time_s = np.arange(WINDOW_SIZE, dtype=dtype) / np.float32(Fs_ASSUMED)
    t_center = np.float32((WINDOW_SIZE - 1) / (2.0 * Fs_ASSUMED))
    time_s = (raw_time_s - t_center).astype(dtype)

    M1 = np.zeros((WINDOW_SIZE, 4 * len(ALL_FREQUENCIES_HZ)), dtype=dtype)
    for i, f in enumerate(ALL_FREQUENCIES_HZ):
        f_val = np.float32(f)
        wt = np.float32(2.0 * np.pi) * f_val * time_s
        sin_wt = np.sin(wt).astype(dtype)
        cos_wt = np.cos(wt).astype(dtype)
        M1[:, 4 * i] = sin_wt
        M1[:, 4 * i + 1] = cos_wt
        M1[:, 4 * i + 2] = time_s * sin_wt
        M1[:, 4 * i + 3] = time_s * cos_wt

    PRE['C1'] = np.linalg.pinv(M1).astype(dtype)
    PRE['time_s'] = time_s
    PRE['two_pi'] = np.float32(2.0 * np.pi)
    PRE['sqrt2'] = np.float32(np.sqrt(2.0))

def estimate_real_memory_kb(*matrices) -> float:
    total_bytes = sum(m.nbytes for m in matrices if isinstance(m, np.ndarray))
    return total_bytes / 1024.0

def process_window_accurate(S_window: np.ndarray, window_idx: int, start_sample: int, theory_dict=None) -> Dict[str, Any]:
    start_time = time.perf_counter()
    dtype = np.float32

    if S_window.dtype != dtype:
        S_typed = S_window.astype(dtype)
    else:
        S_typed = S_window

    t_rel = PRE['time_s']

    # --- 阶段 1：初估频率 ---
    P1 = (PRE['C1'] @ S_typed).astype(dtype)
    f_init = []
    for i, f in enumerate(ALL_FREQUENCIES_HZ):
        a, b, c, d = P1[4 * i: 4 * i + 4]
        denom = a ** 2 + b ** 2
        delta_f = (a * d - b * c) / (PRE['two_pi'] * denom) if denom > 1e-12 else np.float32(0.0)
        f_init.append(np.float32(ALL_FREQUENCIES_HZ[i]) + np.clip(delta_f, np.float32(-MAX_FREQ_DEVIATION_HZ), np.float32(MAX_FREQ_DEVIATION_HZ)))

    # --- 阶段 2：迭代精修频率 ---
    new_f_nom = [np.float32(f_init[i]) if f >= 50.0 else np.float32(f) for i, f in enumerate(ALL_FREQUENCIES_HZ)]
    
    SUBSAMPLE = 3
    t_rel_sub = t_rel[::SUBSAMPLE]
    S_typed_sub = S_typed[::SUBSAMPLE]
    WINDOW_SIZE_SUB = len(t_rel_sub)
    
    M2 = np.zeros((WINDOW_SIZE_SUB, 4 * len(new_f_nom)), dtype=dtype)
    for i, f in enumerate(new_f_nom):
        wt = PRE['two_pi'] * f * t_rel_sub
        sin_wt = np.sin(wt).astype(dtype)
        cos_wt = np.cos(wt).astype(dtype)
        M2[:, 4 * i] = sin_wt
        M2[:, 4 * i + 1] = cos_wt
        M2[:, 4 * i + 2] = t_rel_sub * sin_wt
        M2[:, 4 * i + 3] = t_rel_sub * cos_wt

    P2 = c_lstsq(M2, S_typed_sub)

    f_final = []
    for i in range(len(new_f_nom)):
        a, b, c, d = P2[4 * i: 4 * i + 4]
        denom = a ** 2 + b ** 2
        delta_f = (a * d - b * c) / (PRE['two_pi'] * denom) if denom > 1e-12 else np.float32(0.0)
        f_final.append(new_f_nom[i] + np.clip(delta_f, np.float32(-MAX_FREQ_DEVIATION_HZ), np.float32(MAX_FREQ_DEVIATION_HZ)))

    # --- 阶段 3：最终拟合与延迟计算 ---
    X = np.empty((WINDOW_SIZE_SUB, 2 * len(f_final)), dtype=dtype)
    for i, f in enumerate(f_final):
        wt = PRE['two_pi'] * f * t_rel_sub
        X[:, 2 * i] = np.sin(wt).astype(dtype)
        X[:, 2 * i + 1] = np.cos(wt).astype(dtype)

    theta = c_lstsq(X, S_typed_sub)

    t_center_absolute = np.float32((start_sample + (WINDOW_SIZE - 1) / 2.0) / Fs_ASSUMED)
    mem_kb = estimate_real_memory_kb(M2, P2, X, P1, S_typed)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    res = {"窗口": window_idx, "耗时(ms)": elapsed_ms, "内存占用(kb)": mem_kb}

    for i, f_nom in enumerate(ALL_FREQUENCIES_HZ):
        f_val = f_final[i]
        a = theta[2 * i]
        b = theta[2 * i + 1]

        A_rms_est = np.float32(np.sqrt(a ** 2 + b ** 2)) / PRE['sqrt2']
        res[f"{int(f_nom)}Hz估算有效值(mv)"] = A_rms_est

        if f_nom in DELAY_TARGET_FREQS:
            # 窗口计算延迟
            phi_center = np.float32(-np.arctan2(b, a))
            phi_absolute = phi_center + PRE['two_pi'] * f_val * np.float32(t_center_absolute)
            y_ms = (phi_absolute / (PRE['two_pi'] * f_val)) * np.float32(1000.0)
            period_ms = np.float32(1000.0) / f_val
            window_calculated_delay = y_ms % period_ms
            res[f"{int(f_nom)}Hz窗口计算延迟(ms)"] = window_calculated_delay
            
            # --- 窗口理论延迟：直接从理论值计算！---
            if theory_dict and f_nom in theory_dict:
                theory_signal = theory_dict[f_nom]
                # 用理论值计算窗口中心处的理论延迟
                center_sample = start_sample + (WINDOW_SIZE - 1) // 2
                if center_sample < len(theory_signal):
                    # 从理论值中计算理论延迟
                    # 先计算理论值在窗口中心附近的相位
                    # 用窗口中心和前后一个点计算更准确的相位
                    if center_sample > 0 and center_sample < len(theory_signal) - 1:
                        # 用数值微分计算导数
                        val_c = theory_signal[center_sample]
                        val_prev = theory_signal[center_sample - 1]
                        val_next = theory_signal[center_sample + 1]
                        dt = 1.0 / Fs_ASSUMED
                        # 中心差分求导数
                        dy_dt = (val_next - val_prev) / (2 * dt)
                        w = 2 * np.pi * f_nom
                        # 计算相位
                        A_theory = np.sqrt(val_c**2 + (dy_dt / w)**2)
                        if A_theory > 1e-6:
                            # phi满足 val_c = A*sin(w*t_center + phi)
                            # dy_dt = A*w*cos(w*t_center + phi)
                            phi = np.arctan2(val_c * w, dy_dt)
                            # 延迟为 phi / (2*pi*f_nom) * 1000 ms
                            theory_delay_ms = (phi / (2 * np.pi * f_nom)) * 1000
                            # 调整到 [0, 周期) 范围内
                            theory_period_ms = 1000.0 / f_nom
                            theory_delay_mod = theory_delay_ms % theory_period_ms
                            res[f"{int(f_nom)}Hz窗口理论延迟(ms)"] = theory_delay_mod
                        else:
                            res[f"{int(f_nom)}Hz窗口理论延迟(ms)"] = 0.0
                    else:
                        res[f"{int(f_nom)}Hz窗口理论延迟(ms)"] = 0.0
                else:
                    res[f"{int(f_nom)}Hz窗口理论延迟(ms)"] = 0.0
            else:
                res[f"{int(f_nom)}Hz窗口理论延迟(ms)"] = 0.0
            
            # 计算误差
            theory_delay = res.get(f"{int(f_nom)}Hz窗口理论延迟(ms)", 0.0)
            error = window_calculated_delay - theory_delay
            if error > period_ms / 2:
                error -= period_ms
            elif error < -period_ms / 2:
                error += period_ms
            res[f"{int(f_nom)}Hz计算误差(ms)"] = error

        res[f"{int(f_nom)}Hz监测频率(Hz)"] = f_val
        true_rms = TARGET_RMS_VALUES.get(f_nom, 0.0)
        if true_rms == 0:
            res[f"{int(f_nom)}Hz偏差百分比(%)"] = np.float32(0.0)
        else:
            res[f"{int(f_nom)}Hz偏差百分比(%)"] = (A_rms_est - np.float32(true_rms)) / np.float32(true_rms) * np.float32(100.0)

    return res

def main():
    precompute_assets()

    try:
        df_input = pd.read_excel(excel_filename_input)
        full_signal = df_input[COLUMN_NOISY].values.astype(np.float32)
        print(f"Loaded successfully, size: {len(full_signal)}")
        
        # 加载理论值
        theory_dict = {}
        for freq in [4, 8, 50, 128]:
            col_name = f'{freq}Hz理论值'
            if col_name in df_input.columns:
                theory_dict[freq] = df_input[col_name].values
                theory_dict[float(freq)] = df_input[col_name].values
        print(f"Loaded theory values for frequencies: {list(theory_dict.keys())}")
        
    except Exception as e:
        print(f"Read failed: {e}")
        import traceback
        traceback.print_exc()
        return

    global_dc_bias = np.mean(full_signal)
    full_signal_filtered = full_signal - global_dc_bias
    print(f"Detected and filtered global DC bias: {global_dc_bias:.4f} mv")

    all_window_results = []
    step = int(Fs_ASSUMED * (SLIDE_STEP_MS / 1000.0))

    start_all = time.perf_counter()

    for idx, start in enumerate(range(0, len(full_signal_filtered) - WINDOW_SIZE + 1, step)):
        S_window_filtered = full_signal_filtered[start: start + WINDOW_SIZE]
        
        try:
            result = process_window_accurate(S_window_filtered, idx + 1, start, theory_dict=theory_dict)
            all_window_results.append(result)
        except Exception as e:
            print(f"Window {idx + 1} error: {e}")
            import traceback
            traceback.print_exc()

    end_all = time.perf_counter()
    print(f"Average time per window: {(end_all - start_all) * 1000 / len(all_window_results):.2f} ms")

    df_output = pd.DataFrame(all_window_results)

    cols = ["窗口"]
    for f in ALL_FREQUENCIES_HZ:
        f = int(f)
        cols.extend([f"{f}Hz估算有效值(mv)", f"{f}Hz偏差百分比(%)", f"{f}Hz窗口计算延迟(ms)", f"{f}Hz窗口理论延迟(ms)", f"{f}Hz计算误差(ms)", f"{f}Hz监测频率(Hz)"])
    cols.extend(["耗时(ms)", "内存占用(kb)"])

    df_output[[c for c in cols if c in df_output.columns]].to_excel(excel_filename_output, index=False)
    print(f"\nFinished processing, results saved to {excel_filename_output}")
    
    # 打印误差统计
    print("\n误差统计:")
    for f in [128]:
        error_col = f"{f}Hz计算误差(ms)"
        if error_col in df_output.columns:
            errors = df_output[error_col].dropna()
            print(f"\n{f}Hz:")
            print(f"  样本数: {len(errors)}")
            print(f"  平均误差: {errors.mean():.6f} ms")
            print(f"  误差标准差: {errors.std():.6f} ms")
            print(f"  误差绝对值小于0.1ms的比例: {(errors.abs() < 0.1).mean() * 100:.2f}%")
            print(f"  最大误差绝对值: {errors.abs().max():.6f} ms")

if __name__ == '__main__':
    main()
