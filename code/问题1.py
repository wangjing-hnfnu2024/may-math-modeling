import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import platform, warnings, os
warnings.filterwarnings('ignore')
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ★修正1：在模块顶层统一定义代码文件所在目录，供绘图、保存结果使用
sd = os.path.dirname(os.path.abspath(__file__))

def find_file(fn):
    if os.path.exists(fn):
        return fn
    p = os.path.join('cleaned_results', fn)
    if os.path.exists(p):
        return p
    p2 = os.path.join(sd, fn)          # 用模块级 sd，避免内部重复定义
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"找不到文件 '{fn}'")

# ★修正2：去掉文件名末尾的空格
fn = "../data/附件1：两组位移时序数据-问题1.xlsx"
df = pd.read_excel(fn)
print("=" * 65)
print("  增量空间鲁棒多项式回归 — 数据A校正模型")
print("=" * 65)
print(f"\n[数据读取] 路径: {fn}")
print(f"[数据读取] 共 {len(df)} 行 × {len(df.columns)} 列")
lx = ['数据A', '光纤位移计', '光纤位移计原始读数', 'xv',
      '数据A（光纤位移计原始读数）', '数据A（光纤位移计）', '光纤']
ly = ['数据B', '振弦式位移计', '振弦式位移计基准读数', 'yv',
      '数据B（振弦式位移计基准读数）', '数据B（振弦式位移计）', '振弦']
cx = cy = None
for c in df.columns:
    cn = str(c).strip()
    if cx is None:
        for m in lx:
            if m in cn:
                cx = c
                break
    if cy is None:
        for m in ly:
            if m in cn:
                cy = c
                break
if cx is None or cy is None:
    nc = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(nc) >= 2:
        cx = cx or nc[0]
        cy = cy or nc[1]
        print(f"[列名识别] 回退使用前两列: '{cx}', '{cy}'")
xv = df[cx].values.astype(float)
yv = df[cy].values.astype(float)
N = len(xv)
dx = np.diff(xv)
dy = np.diff(yv)
M = len(dx)
dx2d = dx.reshape(-1, 1)
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('hub', HuberRegressor(epsilon=1.345, max_iter=2000, tol=1e-6))
])
pipe.fit(dx2d, dy)
hb = pipe.named_steps['hub']
t0, t1, t2, t3 = hb.intercept_, *hb.coef_
print("\n" + "=" * 65)
print("  全量训练结果 — 增量校正映射方程")
print("=" * 65)
print(f"\n  Δŷt = {t0:.6f} + ({t1:.6f})·Δxt"
      f" + ({t2:.6f})·Δxt² + ({t3:.6f})·Δxt³")
dy_pred = pipe.predict(dx2d)
r2_d = r2_score(dy, dy_pred)
K = 5
tsz = M // K
idx = np.arange(M)
r2_f, mae_f = [], []
print("\n" + "=" * 65)
print("  5折时序交叉验证（增量空间滚动窗口）")
print("=" * 65)
print(f"\n  {'折次':^6}{'训练集':^14}{'测试集':^14}{'R²':>12}{'MAE (mm)':>12}")
print("  " + "-" * 58)
for k in range(1, K + 1):
    ts = k * tsz
    te = (k + 1) * tsz if k < K else M
    tri = idx[:ts]
    tei = idx[ts:te]
    if len(tei) == 0 or len(tri) == 0:
        continue
    Xtr, ytr = dx[tri].reshape(-1, 1), dy[tri]
    Xte, yte = dx[tei].reshape(-1, 1), dy[tei]
    pk = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('hub', HuberRegressor(epsilon=1.345, max_iter=2000, tol=1e-6))
    ])
    pk.fit(Xtr, ytr)
    ypk = pk.predict(Xte)
    r2k = r2_score(yte, ypk)
    maek = mean_absolute_error(yte, ypk)
    r2_f.append(r2k)
    mae_f.append(maek)
    print(f"  第{k}折  {len(tri):>6} 样本  {len(tei):>6} 样本"
          f"  {r2k:>10.6f}  {maek:>10.6f}")
print("  " + "-" * 58)
print(f"  {'均值':^6}{'':^14}{'':^14}{np.mean(r2_f):>10.6f}"
      f"  {np.mean(mae_f):>10.6f}")
print(f"  {'标准差':^6}{'':^14}{'':^14}{np.std(r2_f):>10.6f}"
      f"  {np.std(mae_f):>10.6f}")
yc = np.zeros(N)
yc[0] = yv[0]
for i in range(1, N):
    yc[i] = yc[i - 1] + dy_pred[i - 1]
r2_a = r2_score(yv, yc)
mae_a = mean_absolute_error(yv, yc)
rmse_a = np.sqrt(np.mean((yv - yc) ** 2))
maxe = np.max(np.abs(yv - yc))
if HAS_PLT:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        osn = platform.system()
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] if osn == 'Windows' else ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        C1, C2, C3, C4 = '#4C72B0', '#DD8452', '#C0C4CC', '#C44E52'
        def cln(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#666')
            ax.spines['bottom'].set_color('#666')
        ta = np.arange(N)
        si = np.argsort(dx)
        err = yv - yc
        ep = np.where(err > 0, err, 0)
        en = np.where(err < 0, err, 0)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(xv, yv, s=15, alpha=0.5, color=C1, edgecolors='none')
        ax1.set_title('原始传感器数据对应关系', fontweight='bold')
        cln(ax1)
        fig1.savefig(os.path.join(sd, '图1_原始数据散点.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(dx, dy, s=10, alpha=0.3, color=C3, edgecolors='none')
        ax2.plot(dx[si], dy_pred[si], color=C2, linewidth=2.8)
        ax2.set_title(f'增量空间拟合效果 (R² = {r2_d:.4f})', fontweight='bold')
        cln(ax2)
        fig2.savefig(os.path.join(sd, '图2_增量空间拟合.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        fig3, ax3 = plt.subplots(figsize=(10, 5.5))
        ax3.plot(ta, yv, color=C1, linewidth=1.8, label='数据B (基准)')
        ax3.plot(ta, yc, color=C2, linewidth=2.2, linestyle='--', label='校正后数据A')
        ax3.set_title(f'绝对位移校正对比 (MAE = {mae_a:.2f} mm)', fontweight='bold')
        cln(ax3)
        ax3.legend()
        fig3.savefig(os.path.join(sd, '图3_绝对位移时序对比.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig3)
        fig4, ax4 = plt.subplots(figsize=(10, 5.5))
        ax4.fill_between(ta, 0, ep, color=C1, alpha=0.5, label='正残差')
        ax4.fill_between(ta, 0, en, color=C4, alpha=0.5, label='负残差')
        ax4.axhline(0, color='#333', linewidth=1.2)
        ax4.set_title(f'校正残差分布 (最大误差 = {maxe:.2f} mm)', fontweight='bold')
        cln(ax4)
        ax4.legend()
        fig4.savefig(os.path.join(sd, '图4_校正残差分布.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig4)
        print("\n[可视化] 四张图片已保存")
    except Exception as e:
        print(f"\n[警告] 图片生成出错: {e}")
print("\n" + "=" * 65)
print("  问题1 数据校正结果表 (前5个点)")
print("=" * 65)
print(f"  {'序号':<6}| {'校正前数据x (mm)':<22}| {'校正后数据y (mm)':<22}")
print("  " + "-" * 55)
for i in range(5):
    print(f"  {i+1:<6}| {xv[i]:<22.4f}| {yc[i]:<22.4f}")
rdf = pd.DataFrame({
    cx: xv,
    cy: yv,
    '校正后数据A(mm)': np.round(yc, 4)
})
sp = os.path.join(sd, 'calibrated_output.xlsx')
rdf.to_excel(sp, index=False)
print(f"\n[保存] 校正结果已保存至: {sp}")
print("=" * 65)
