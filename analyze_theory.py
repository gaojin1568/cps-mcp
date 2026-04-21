import pandas as pd
import numpy as np

df_input = pd.read_excel('/workspace/模拟数据.xlsx')

Fs_ASSUMED = 1600.0
freqs = [4, 8, 50, 128]

print("详细分析理论值的构造方式:")
print("=" * 80)

for freq in freqs:
    col_name = f'{freq}Hz理论值'
    if col_name not in df_input.columns:
        continue
    
    theory = df_input[col_name].values
    t = np.arange(len(theory)) / Fs_ASSUMED
    
    # 计算相位和延迟
    amplitude = np.sqrt(np.mean(theory ** 2) * 2)
    print(f"\n{freq}Hz:")
    print(f"  估计幅度: {amplitude:.6f}")
    
    # 尝试拟合相位关系
    # 查看前几个点的变化
    print(f"  前5点: {theory[:5]}")
    
    # 计算理论延迟 - 从第0点的相位反推
    if freq == 128:
        # 对于128Hz，我们来手动计算一下理论相位
        # 假设理论值是 A * sin(2*pi*f*t + phi)
        # 我们可以从第0点和导数来计算
        if len(theory) >= 2:
            dt = 1.0 / Fs_ASSUMED
            dy_dt = (theory[1] - theory[0]) / dt
            val0 = theory[0]
            
            # 计算相位 phi，满足 val0 = A*sin(phi), dy_dt = A*w*cos(phi)
            w = 2 * np.pi * freq
            A_est = np.sqrt(val0**2 + (dy_dt / w)**2)
            phi_est = np.arctan2(val0 * w, dy_dt)
            
            print(f"  从第0点估计:")
            print(f"    A: {A_est:.6f}")
            print(f"    phi: {phi_est:.6f} rad")
            print(f"    相位对应的延迟 (ms): {(phi_est / (2*np.pi*freq)) * 1000:.6f}")
    
    # 看看滑动窗口中心对应的理论值的延迟应该怎么算
    print(f"\n  窗口分析:")
    WINDOW_SIZE = int(Fs_ASSUMED * 0.5)
    SLIDE_STEP_MS = 20.0
    step = int(Fs_ASSUMED * (SLIDE_STEP_MS / 1000.0))
    
    for window_idx in range(1, 4):
        start_sample = (window_idx - 1) * step
        center_sample = start_sample + (WINDOW_SIZE - 1) // 2
        center_val = theory[center_sample] if center_sample < len(theory) else 0
        
        # 计算窗口中心相对于第0点的理论延迟
        # 窗口中心的绝对时间
        t_center = (start_sample + (WINDOW_SIZE - 1) / 2.0) / Fs_ASSUMED
        
        # 如果理论值的初始延迟是固定的，那么窗口中心的理论延迟应该是：
        # 初始延迟 + t_center 的时间？
        
        print(f"  窗口 {window_idx}:")
        print(f"    start_sample: {start_sample}, center_sample: {center_sample}")
        print(f"    t_center: {t_center*1000:.3f}ms")
        print(f"    center_val: {center_val:.6f}")
