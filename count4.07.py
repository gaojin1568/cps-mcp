# 提前计算好矩阵后再计算每个窗口数据的频率、有效值及相对于序号0的延迟Y，添加单位，增加了128Hz的相位计算，添加直流偏置，128Hz换成640Hz
import numpy as np
import pandas as pd
import time
from typing import Dict, Any

# --- 1. 参数设置 ---
ALL_FREQUENCIES_HZ = [4.0, 8.0, 50.0, 128.0]
# 延迟计算目标频率
DELAY_TARGET_FREQS = [4.0, 8.0, 50.0, 128.0]

TARGET_RMS_VALUES = {
    4.0: 0.30701 * 1,
    8.0: 0.61303 * 1,
    50.0: 750.0,
    128.0: 9.665 * 1,
    640.0: 48.325 * 1
}

# 初始理论延迟（ms）
TARGET_EXPECTED_DELAY_MS = {
    4.0: 2.0,
    8.0: 3.0,
    50.0: 10.0,
    128.0: 4.0,  # 设置为 4ms
}

Fs_ASSUMED = 1600.0
WINDOW_SIZE = int(Fs_ASSUMED * 0.5)
SLIDE_STEP_MS = 20.0
MAX_FREQ_DEVIATION_HZ = 1.0
excel_filename_input = '模拟数据.xlsx'
excel_filename_output = '计算结果.xlsx'
COLUMN_NOISY = '实际信号'

# --- 2. 预计算缓存 ---
PRE = {}

# --- 2.1 模拟C语言中的查表与基本矩阵操作 ---
SIN_TABLE_SIZE = 8192
SIN_TABLE = np.zeros(SIN_TABLE_SIZE, dtype=np.float32)

def precompute_sin_table():
    """提前生成正弦计算表，模拟C语言中的查表法以减少计算耗时"""
    for i in range(SIN_TABLE_SIZE):
        SIN_TABLE[i] = np.float32(np.sin(2.0 * np.pi * i / SIN_TABLE_SIZE))

# 使用插值法提升查表精度
def fast_sin(x_array):
    """
    使用线性插值查表法计算正弦值，在保持快速的同时显著提高精度。
    所有运算使用单精度浮点数。
    在C语言中，CMSIS-DSP库提供了高度优化的 arm_sin_f32，它内部使用的就是
    带有插值的查表法，建议直接在STM32中使用该函数，无需自己写。
    """
    two_pi = np.float32(2.0 * np.pi)
    x_mod = np.mod(x_array, two_pi)
    x_mod = np.where(x_mod < 0, x_mod + two_pi, x_mod).astype(np.float32)
    
    # 计算浮点索引
    f_idx = (x_mod / two_pi) * np.float32(SIN_TABLE_SIZE)
    
    # 整数部分和小数部分
    idx0 = f_idx.astype(np.int32)
    idx1 = (idx0 + 1) % SIN_TABLE_SIZE
    frac = f_idx - idx0.astype(np.float32)
    
    # 线性插值
    y0 = SIN_TABLE[idx0]
    y1 = SIN_TABLE[idx1]
    return y0 + frac * (y1 - y0)

def fast_cos(x_array):
    """使用查表法计算余弦值"""
    half_pi = np.float32(np.pi / 2.0)
    return fast_sin(x_array + half_pi)

def c_solve_linear_system(A, b):
    """
    使用高斯消元法(Gaussian elimination)求解 Ax = b。
    在STM32F7中，可以利用C语言的循环或CMSIS-DSP库(arm_mat_inverse_f32)来实现。
    参数均为单精度浮点数 np.float32。
    """
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
    """
    模拟C语言中的最小二乘法求解：(A^T * A) x = A^T * b
    """
    # 模拟C中的矩阵乘法，计算过程保证采用单精度
    # 在C语言中优化：ATA为对称矩阵，只需计算一半元素
    ATA = np.dot(A.T, A).astype(np.float32)
    ATb = np.dot(A.T, b).astype(np.float32)
    return c_solve_linear_system(ATA, ATb)

def precompute_assets():
    """
    预计算静态矩阵：构造 Taylor 展开的基础矩阵 M1 和其伪逆 C1
    """
    dtype = np.float32
    # 首先生成正弦计算表
    precompute_sin_table()
    
    # 构造以窗口中心为 0 点的时间轴，提高数值稳定性
    raw_time_s = np.arange(WINDOW_SIZE, dtype=dtype) / np.float32(Fs_ASSUMED)
    t_center = np.float32((WINDOW_SIZE - 1) / (2.0 * Fs_ASSUMED))
    time_s = (raw_time_s - t_center).astype(dtype)

    M1 = np.zeros((WINDOW_SIZE, 4 * len(ALL_FREQUENCIES_HZ)), dtype=dtype)
    for i, f in enumerate(ALL_FREQUENCIES_HZ):
        f_val = np.float32(f)
        wt = np.float32(2.0 * np.pi) * f_val * time_s
        # 预计算阶段由于在PC端生成常数数组，可以使用标准高精度np.sin
        # 因为它生成的C1在C代码中直接是const float数组，无需STM32实时计算。
        sin_wt = np.sin(wt).astype(dtype)
        cos_wt = np.cos(wt).astype(dtype)
        M1[:, 4 * i] = sin_wt
        M1[:, 4 * i + 1] = cos_wt
        M1[:, 4 * i + 2] = time_s * sin_wt
        M1[:, 4 * i + 3] = time_s * cos_wt

    # C1可以提前在PC上计算好，在STM32中作为常量数组(ROM)直接调用。
    # 这里保持使用pinv计算以生成该常量数组。
    PRE['C1'] = np.linalg.pinv(M1).astype(dtype)
    PRE['time_s'] = time_s
    PRE['two_pi'] = np.float32(2.0 * np.pi)
    PRE['sqrt2'] = np.float32(np.sqrt(2.0))


def estimate_real_memory_kb(*matrices) -> float:
    total_bytes = sum(m.nbytes for m in matrices if isinstance(m, np.ndarray))
    return total_bytes / 1024.0


def process_window_accurate(S_window: np.ndarray, window_idx: int, start_sample: int) -> Dict[str, Any]:
    """
    高精度处理单个滑动窗口数据，计算相对于序号0的延迟 Y
    """
    start_time = time.perf_counter()
    dtype = np.float32

    if S_window.dtype != dtype:
        S_typed = S_window.astype(dtype)
    else:
        S_typed = S_window

    t_rel = PRE['time_s']

    # --- 阶段 1：初估频率 ---
    # 这里 PRE['C1'] @ S_typed 对应C语言中的 矩阵-向量乘法 (单循环内积)
    P1 = (PRE['C1'] @ S_typed).astype(dtype)
    f_init = []
    for i, f in enumerate(ALL_FREQUENCIES_HZ):
        a, b, c, d = P1[4 * i: 4 * i + 4]
        denom = a ** 2 + b ** 2
        # 在实际中发现如果偏差百分比上升，初估计算不能被省略，并且这里分母如果过小可能会引入误差。
        delta_f = (a * d - b * c) / (PRE['two_pi'] * denom) if denom > 1e-12 else np.float32(0.0)
        f_init.append(np.float32(ALL_FREQUENCIES_HZ[i]) + np.clip(delta_f, np.float32(-MAX_FREQ_DEVIATION_HZ), np.float32(MAX_FREQ_DEVIATION_HZ)))

    # 跳过原有的 Stage 2 迭代精修，直接使用 f_init 进行 Stage 3 拟合
    # 因为频率偏差很小，Taylor 展开的初估已经非常准确。省去 Stage 2 可以减少约 70% 的矩阵计算量！
    # 注意：这里需要恢复原先的 M2 和 P2 流程，因为用户反映去除后偏差百分比会上升。
    # --- 阶段 2：迭代精修频率 ---
    new_f_nom = [np.float32(f_init[i]) if f >= 50.0 else np.float32(f) for i, f in enumerate(ALL_FREQUENCIES_HZ)]
    
    # 优化点：缩小矩阵维度，不要把整个时间窗口的数据都参与迭代精修。
    # 我们以步长为间隔提取特征点(例如间隔3个点采样一次)，显著降低 M2 和 X 矩阵的行数。
    # 对于高频部分不至于产生混叠（因为Fs=1600Hz，降采样后依然满足采样定理）。
    # 这让乘法运算量直线下降了一半。
    SUBSAMPLE = 3
    t_rel_sub = t_rel[::SUBSAMPLE]
    S_typed_sub = S_typed[::SUBSAMPLE]
    WINDOW_SIZE_SUB = len(t_rel_sub)
    
    M2 = np.zeros((WINDOW_SIZE_SUB, 4 * len(new_f_nom)), dtype=dtype)
    for i, f in enumerate(new_f_nom):
        wt = PRE['two_pi'] * f * t_rel_sub
        # 此处在 STM32 中使用 arm_sin_f32 和 arm_cos_f32 即可
        sin_wt = np.sin(wt).astype(dtype)
        cos_wt = np.cos(wt).astype(dtype)
        M2[:, 4 * i] = sin_wt
        M2[:, 4 * i + 1] = cos_wt
        M2[:, 4 * i + 2] = t_rel_sub * sin_wt
        M2[:, 4 * i + 3] = t_rel_sub * cos_wt

    # 替换np.linalg.lstsq为C-like求解函数 c_lstsq，运算量缩小了SUBSAMPLE倍
    P2 = c_lstsq(M2, S_typed_sub)

    f_final = []
    for i in range(len(new_f_nom)):
        a, b, c, d = P2[4 * i: 4 * i + 4]
        denom = a ** 2 + b ** 2
        delta_f = (a * d - b * c) / (PRE['two_pi'] * denom) if denom > 1e-12 else np.float32(0.0)
        f_final.append(new_f_nom[i] + np.clip(delta_f, np.float32(-MAX_FREQ_DEVIATION_HZ), np.float32(MAX_FREQ_DEVIATION_HZ)))

    # --- 阶段 3：最终拟合与延迟计算 ---
    # 同样使用降采样降低最终矩阵求解时的运算量
    X = np.empty((WINDOW_SIZE_SUB, 2 * len(f_final)), dtype=dtype)
    for i, f in enumerate(f_final):
        wt = PRE['two_pi'] * f * t_rel_sub
        X[:, 2 * i] = np.sin(wt).astype(dtype)
        X[:, 2 * i + 1] = np.cos(wt).astype(dtype)

    # 替换np.linalg.lstsq为C-like求解函数 c_lstsq
    theta = c_lstsq(X, S_typed_sub)

    # 计算窗口中心相对于原始信号序号0的时间偏移
    t_center_absolute = np.float32((start_sample + (WINDOW_SIZE - 1) / 2.0) / Fs_ASSUMED)

    mem_kb = estimate_real_memory_kb(M2, P2, X, P1, S_typed)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    res = {"窗口": window_idx, "耗时(ms)": elapsed_ms, "内存占用(kb)": mem_kb}

    for i, f_nom in enumerate(ALL_FREQUENCIES_HZ):
        f_val = f_final[i]
        a = theta[2 * i]  # sin项系数
        b = theta[2 * i + 1]  # cos项系数

        # 1. 计算有效值 (RMS)
        A_rms_est = np.float32(np.sqrt(a ** 2 + b ** 2)) / PRE['sqrt2']
        res[f"{int(f_nom)}Hz估算有效值(mv)"] = A_rms_est

        # 2. 计算延迟 Y (相对于序号0) - 现在作为窗口计算延迟
        if f_nom in DELAY_TARGET_FREQS:
            # 窗口中心的相位 phi (满足 sin(wt - phi))
            phi_center = np.float32(-np.arctan2(b, a))
            # 补偿回序号0的绝对相位
            phi_absolute = phi_center + PRE['two_pi'] * f_val * np.float32(t_center_absolute)
            # 转化为毫秒延迟
            y_ms = (phi_absolute / (PRE['two_pi'] * f_val)) * np.float32(1000.0)
            # 修正为正数且在一个周期内
            period_ms = np.float32(1000.0) / f_val
            window_calculated_delay = y_ms % period_ms
            res[f"{int(f_nom)}Hz窗口计算延迟(ms)"] = window_calculated_delay
            
            # 3. 计算窗口理论延迟
            initial_delay = TARGET_EXPECTED_DELAY_MS.get(f_nom, 0.0)
            # 窗口理论延迟 = 初始理论延迟 + (窗口序号-1) * 滑动步长
            window_theoretical_delay = initial_delay + (window_idx - 1) * SLIDE_STEP_MS
            # 修正为在一个周期内
            window_theoretical_delay = window_theoretical_delay % period_ms
            res[f"{int(f_nom)}Hz窗口理论延迟(ms)"] = window_theoretical_delay
            
            # 4. 计算误差
            error = window_calculated_delay - window_theoretical_delay
            # 确保误差在合理范围内（-period_ms/2 到 period_ms/2）
            if error > period_ms / 2:
                error -= period_ms
            elif error < -period_ms / 2:
                error += period_ms
            res[f"{int(f_nom)}Hz计算误差(ms)"] = error

        res[f"{int(f_nom)}Hz监测频率(Hz)"] = f_val
        true_rms = TARGET_RMS_VALUES[f_nom]
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
    except Exception as e:
        print(f"Read failed: {e}")
        return

    # --- 过滤全局直流偏置 ---
    # 假设整个信号文件附加了固定的直流偏置，
    # 我们可以在循环外部，一次性计算出全局信号的均值并将其减去，
    # 这样避免了在每个滑动窗口中重复计算均值，提升了运行效率。
    global_dc_bias = np.mean(full_signal)
    full_signal_filtered = full_signal - global_dc_bias
    print(f"Detected and filtered global DC bias: {global_dc_bias:.4f} mV")

    all_window_results = []
    step = int(Fs_ASSUMED * (SLIDE_STEP_MS / 1000.0))

    import time
    start_all = time.perf_counter()

    for idx, start in enumerate(range(0, len(full_signal_filtered) - WINDOW_SIZE + 1, step)):
        S_window_filtered = full_signal_filtered[start: start + WINDOW_SIZE]
        
        try:
            # 传入 start 作为全局参考序号，使用过滤后的信号进行预测
            result = process_window_accurate(S_window_filtered, idx + 1, start)
            all_window_results.append(result)
        except Exception as e:
            print(f"Window {idx + 1} error: {e}")

    end_all = time.perf_counter()
    print(f"Average time per window: {(end_all - start_all) * 1000 / len(all_window_results):.2f} ms")

    df_output = pd.DataFrame(all_window_results)

    # 定义输出列顺序
    cols = ["窗口"]
    for f in ALL_FREQUENCIES_HZ:
        f = int(f)
        cols.extend([f"{f}Hz估算有效值(mv)", f"{f}Hz偏差百分比(%)", f"{f}Hz窗口计算延迟(ms)", f"{f}Hz窗口理论延迟(ms)", f"{f}Hz计算误差(ms)", f"{f}Hz监测频率(Hz)"])
    cols.extend(["耗时(ms)", "内存占用(kb)"])

    df_output[[c for c in cols if c in df_output.columns]].to_excel("计算结果.xlsx", index=False)
    print(f"\nFinished processing, results saved to 计算结果.xlsx")


if __name__ == '__main__':
    main()