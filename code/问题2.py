import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os, warnings, time
warnings.filterwarnings('ignore')

# ★路径修改1：数据目录指向 ../data
sd = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(sd, '..', 'data')
dp = None
tgt = '附件2'
for d in [data_dir, os.path.join(data_dir, '题目2')]:
    if not os.path.exists(d): continue
    for f in os.listdir(d):
        if tgt.lower() in f.lower() and f.endswith('.xlsx'):
            dp = os.path.join(d, f)
            break
    if dp: break
if not dp:
    raise FileNotFoundError(f"找不到包含 '{tgt}' 的Excel文件！")

# ★路径修改2：结果保存目录指向 ../result 并自动创建
result_dir = os.path.join(sd, '..', 'result')
os.makedirs(result_dir, exist_ok=True)

df = pd.read_excel(dp)
print("=" * 70)
print("  基于改进斜率变点检测的三段式滑坡位移阶段划分模型")
print("=" * 70)
print(f"\n[数据读取] 成功定位路径:\n  {dp}")
ct = cy = None
for c in df.columns:
    cn = str(c).strip()
    if ct is None and any(k in cn for k in ['时间', '编号', '序号', 'time', 'id']):
        ct = c
    if cy is None and any(k in cn for k in ['位移', '累计', 'surface', 'y']):
        cy = c
if ct is None: ct = df.columns[0] 
if cy is None:
    nc = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in nc:
        if c != ct: cy = c; break
print(f"[列名识别] 时间列: '{ct}', 位移列: '{cy}'")
tr = df[ct].values.astype(float)
yr = df[cy].values.astype(float)
if np.allclose(tr, np.arange(1, len(tr)+1)):
    t = np.arange(1, len(tr)+1, dtype=float)
else:
    t = tr
y = yr
N = len(t)
print(f"[数据概览] 总数据点数: {N}, 位移范围: [{y.min():.2f}, {y.max():.2f}] mm")

print("\n" + "-" * 70)
print("  步骤 2.2: 计算局部趋势斜率 (滑动窗口 w=50)")
print("-" * 70)
w = 50
hw = w // 2
ks = np.zeros(N)
for i in range(N):
    s = max(0, i - hw)
    e = min(N, i + hw + 1)
    tw, yw = t[s:e], y[s:e]
    n = len(tw)
    ks[i] = (n * np.sum(tw * yw) - np.sum(tw) * np.sum(yw)) / \
            (n * np.sum(tw**2) - np.sum(tw)**2)
print(f"  斜率序列范围: [{ks.min():.6f}, {ks.max():.6f}] mm/编号")

print("\n  步骤 2.2: 遍历求解最优分割点 τ1, τ2...")
st = time.time()
sk = np.zeros(N + 1)
sk2 = np.zeros(N + 1)
for i in range(N):
    sk[i + 1] = sk[i] + ks[i]
    sk2[i + 1] = sk2[i] + ks[i] ** 2
mc = np.inf
t1, t2 = 0, 0
msl = w 
wss1_arr = np.zeros(N)
for i in range(msl, N - 2 * msl + 1):
    n = i
    s1 = sk[i]
    s2 = sk2[i]
    wss1_arr[i] = s2 - (s1 ** 2) / n
for tau1 in range(msl, N - 2 * msl + 1):
    w1 = wss1_arr[tau1]
    tau2_range = np.arange(tau1 + msl, N - msl + 1)
    
    n2 = tau2_range - tau1
    s_k2 = sk[tau2_range] - sk[tau1]
    s_k2_2 = sk2[tau2_range] - sk2[tau1]
    w2_arr = s_k2_2 - (s_k2 ** 2) / n2
    
    n3 = N - tau2_range
    s_k3 = sk[N] - sk[tau2_range]
    s_k3_2 = sk2[N] - sk2[tau2_range]
    w3_arr = s_k3_2 - (s_k3 ** 2) / n3
    
    total_arr = w1 + w2_arr + w3_arr
    min_idx = np.argmin(total_arr)
    
    if total_arr[min_idx] < mc:
        mc = total_arr[min_idx]
        t1, t2 = tau1, tau2_range[min_idx]
print(f"  搜索完成，耗时: {time.time() - st:.2f} 秒")
print(f"\n  >>> 识别结果：")
print(f"  >>> 节点1（缓慢→加速）: T1 = {t1} (对应时间编号 {t[t1-1]:.0f})")
print(f"  >>> 节点2（加速→快速）: T2 = {t2} (对应时间编号 {t[t2-1]:.0f})")

print("\n" + "-" * 70)
print("  步骤 2.3: 各阶段数学模型拟合")
print("-" * 70)
t_1, y_1 = t[:t1], y[:t1]
t_2, y_2 = t[t1:t2], y[t1:t2]
t_3, y_3 = t[t2:], y[t2:]
c1 = np.polyfit(t_1, y_1, 1)
b1, b0 = c1[0], c1[1]
yf1 = np.polyval(c1, t_1)
r1 = 1 - np.sum((y_1 - yf1)**2) / np.sum((y_1 - np.mean(y_1))**2)
print(f"\n  [阶段Ⅰ] 缓慢匀速变形阶段 (t=1 ~ {t1-1})")
print(f"    模型: y = {b0:.4f} + {b1:.6f} * t")
print(f"    参数: 恒定形变速度 β₁ = {b1:.6f} mm/编号")
print(f"    拟合优度: R² = {r1:.6f}")
c2 = np.polyfit(t_2, y_2, 2)
a2, a1, a0 = c2[0], c2[1], c2[2]
yf2 = np.polyval(c2, t_2)
r2 = 1 - np.sum((y_2 - yf2)**2) / np.sum((y_2 - np.mean(y_2))**2)
print(f"\n  [阶段Ⅱ] 加速变形阶段 (t={t1} ~ {t2-1})")
print(f"    模型: y = {a0:.4f} + {a1:.6f} * t + {a2:.8f} * t²")
print(f"    参数: 形变加速度 α₂ = {a2:.8f} mm/编号² (若>0则证实加速机制)")
print(f"    拟合优度: R² = {r2:.6f}")
def em(t, c0, c1, c2):
    return c0 + c1 * np.exp(c2 * t)
p03 = [y_3[-1], -y_3[-1], 0.001] 
try:
    p3, _ = curve_fit(em, t_3, y_3, p0=p03, maxfev=10000)
    g0, g1, g2 = p3
    yf3 = em(t_3, *p3)
    r3 = 1 - np.sum((y_3 - yf3)**2) / np.sum((y_3 - np.mean(y_3))**2)
    print(f"\n  [阶段Ⅲ] 快速变形阶段 (t={t2} ~ {int(t[-1])})")
    print(f"    模型: y = {g0:.4f} + ({g1:.4f}) * e^({g2:.6f} * t)")
    print(f"    参数: 加速因子 γ₂ = {g2:.6f} (值越大临滑越急促)")
    print(f"    拟合优度: R² = {r3:.6f}")
except RuntimeError:
    print("\n  [阶段Ⅲ] 指数模型拟合失败")
    g2 = None
    yf3 = y_3

print("\n" + "-" * 70)
print("  步骤 2.4.2: 各阶段表面位移平均速度计算")
print("-" * 70)
print("  注：修正了原题意笔误，1个编号间隔 = 10分钟 = 1/6 小时")
ys1, ye1 = yf1[0], yf1[-1]
ys2, ye2 = yf2[0], yf2[-1]
ys3, ye3 = yf3[0], yf3[-1]
dt1 = (t1 - 1) * (1 / 6)
ds1 = ye1 - ys1
v1 = ds1 / dt1 if dt1 > 0 else 0
dt2 = (t2 - t1) * (1 / 6)
ds2 = ye2 - ys2
v2 = ds2 / dt2 if dt2 > 0 else 0
tet = 2233
if g2 is not None:
    ye3_t = em(tet, *p3)
else:
    ye3_t = ye3
dt3 = (tet - t2) * (1 / 6)
ds3 = ye3_t - ys3
v3 = ds3 / dt3 if dt3 > 0 else 0
print(f"\n  【阶段Ⅰ 平均速度 V1】")
print(f"    持续时间 ΔT1 = ({t1} - 1) × (1/6) = {dt1:.2f} 小时")
print(f"    位移增量 ΔS1 = {ye1:.4f} - {ys1:.4f} = {ds1:.4f} mm")
print(f"    V1 = ΔS1 / ΔT1 = {v1:.4f} mm/h")
print(f"\n  【阶段Ⅱ 平均速度 V2】")
print(f"    持续时间 ΔT2 = ({t2} - {t1}) × (1/6) = {dt2:.2f} 小时")
print(f"    位移增量 ΔS2 = {ye2:.4f} - {ys2:.4f} = {ds2:.4f} mm")
print(f"    V2 = ΔS2 / ΔT2 = {v2:.4f} mm/h")
print(f"\n  【阶段Ⅲ 平均速度 V3】(计算至编号 {tet})")
print(f"    持续时间 ΔT3 = ({tet} - {t2}) × (1/6) = {dt3:.2f} 小时")
print(f"    位移增量 ΔS3 = {ye3_t:.4f} - {ys3:.4f} = {ds3:.4f} mm")
print(f"    V3 = ΔS3 / ΔT3 = {v3:.4f} mm/h")

print("\n" + "=" * 70)
print("  核心结果汇总")
print("=" * 70)
print(f"  节点识别: T1 = {t1}, T2 = {t2}")
print(f"  速度递增: V1({v1:.2f}) < V2({v2:.2f}) < V3({v3:.2f}) mm/h")
print("=" * 70)
rdf = pd.DataFrame({
    '时间编号_t': t,
    '原始位移_y': y,
    '局部斜率_k': np.round(ks, 6),
    '阶段划分': np.concatenate([
        np.full(t1, '阶段Ⅰ(缓慢)'),
        np.full(t2 - t1, '阶段Ⅱ(加速)'),
        np.full(N - t2, '阶段Ⅲ(快速)')
    ]),
    '拟合位移_yhat': np.concatenate([yf1, yf2, yf3])
})

# ★路径修改3：结果保存到 ../result/
sp = os.path.join(result_dir, 'stage_division_results.xlsx')
rdf.to_excel(sp, index=False)
print(f"\n[保存] 阶段划分结果已保存至:\n  {sp}")
