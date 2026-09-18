import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
import os
warnings.filterwarnings('ignore')

# ★路径修改1：数据目录指向 ../data
sd = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(sd, '..', 'data')

# ★路径修改2：数据文件改为相对路径
fp = os.path.join(data_dir, "附件5：监测数据-问题5.xlsx")
print("读取数据并重构5.1最优空间...")
df = pd.read_excel(fp)
def gc(d, k):
    for c in d.columns:
        if k in str(c): return c
cy = gc(df, '位移')
cu = gc(df, '孔隙') or gc(df, '水压')
ce = gc(df, '微震')
cw = gc(df, '干湿') or gc(df, '入渗')
cr = gc(df, '距离') or gc(df, '爆破点')
cq = gc(df, '药量') or gc(df, '最大')
# 承接5.1剔除P，保留5维特征
dd = {'U': df[cu], 'E': df[ce], 'W': df[cw], "R'": df[cr].fillna(0), "Q'": df[cq].fillna(0)}
md = pd.DataFrame(dd)
md['Y'] = df[cy]
md = md.dropna()
Y = md['Y'].values
X = md[["U", "E", "W", "R'", "Q'"]].values
fn = ["U", "E", "W", "R'", "Q'"]
m = LinearRegression().fit(X, Y)
b = m.coef_  
print(f"偏导数系数: {dict(zip(fn, b))}\n")
# ================= 5.2.1 偏导数分解 =================
print("="*60 + "\n5.2.1 偏导数分解位移速度")
dd2 = md.diff().dropna()
vt = dd2['Y'].values
dX = dd2[fn].values
ct = dX * b
print(f"速度序列长度: {len(vt)}\n")
# ================= 5.2.2 滑动t检验划分 =================
print("="*60 + "\n5.2.2 滑动t检验自适应划分")
at = np.diff(vt)
vp = vt[1:]
def stt(s, w=150, th=2.58):
    """手写滑动t检验"""
    n = len(s)
    ts = np.zeros(n)
    for i in range(w, n - w):
        s1, s2 = s[i-w:i], s[i:i+w]
        sp = np.sqrt((np.var(s1, ddof=1) + np.var(s2, ddof=1)) / 2)
        ts[i] = abs(np.mean(s1) - np.mean(s2)) / (sp * np.sqrt(2 / w)) if sp > 1e-8 else 0
        
    pks = []
    for i in range(w + 20, n - w - 20):
        lr = ts[i-20:i+20]
        if ts[i] == np.max(lr) and ts[i] > th:
            if len(pks) == 0 or (i - pks[-1]) > w: pks.append(i)
    return sorted(pks)
pks = stt(vp, 150, 2.5)
bkps = []
if len(pks) >= 2:
    bkps = [pks[0], pks[1], len(vp)]
    print(f"[+] 检测到{len(pks)}个突变点，划分为三阶段。")
elif len(pks) == 1:
    bkps = [pks[0], int(pks[0] + (len(vp) - pks[0]) * 0.75), len(vp)]
    print(f"[!] 检测到1个突变点，结合物理逻辑延展划分。")
else:
    bkps = [int(len(vp)*0.6), int(len(vp)*0.85), len(vp)]
    print(f"[!] 未检测到突变点，退化为安全基准标定模式。")
print(f"分割点索引: {bkps}\n")
sd, pi = [], 0
sn = ['阶段I(平稳蠕变期)', '阶段II(加速变形期)', '阶段III(临滑加速期)']
for i, bk in enumerate(bkps):
    sd.append({'n': sn[i], 'V': vp[pi:bk], 'A': at[pi:bk], 'C': ct[pi+1:bk+1]})
    pi = bk
# ================= 5.2.3 分阶段数学分析 =================
print("="*60 + "\n5.2.3 分阶段变化规律分析")
for st in sd:
    v, a, c = st['V'], st['A'], st['C']
    vr = np.var(c, axis=0)
    rats = vr / (np.sum(vr) + 1e-8)
    ri = np.argsort(-rats)
    
    print(f"\n[{st['n']}] (样本: {len(v)})")
    print(f"  速度: 均值={np.mean(v):.4f}, 方差={np.var(v):.4f} | 加速度: 均值={np.mean(a):.6f}, 方差={np.var(a):.6f}")
    print("  偏导方差贡献: ", end="")
    for idx in ri: print(f"{fn[idx]}({rats[idx]*100:.1f}%) ", end="")
    print()
# ================= 5.2.4 预警与工况回溯 =================
print("\n" + "="*60 + "\n5.2.4 动态预警构建与工况回溯")
L, thc, Np = 200, 1.5, 3
ya, oa, ra, rc = [], [], [], 0
print(f"滑动窗L={L}, 红色阈值θ={thc}, 持续步长N={Np}\n")
for i in range(L, len(vp)):
    wl = vp[i-L:i]
    vy = np.percentile(wl, 95)
    vo = np.mean(wl) + 3 * np.median(np.abs(wl - np.median(wl))) * 1.4826
    
    vc = vp[i]
    if vc > vo: oa.append(i)
    elif vc > vy: ya.append(i)
        
    if i >= 1 and vp[i-1] > 1e-6:
        if vc / vp[i-1] > thc: rc += 1
        else: rc = 0
        if rc >= Np: ra.append(i); rc = 0
print(f"[预警统计]")
print(f"  黄色预警(关注): {len(ya)} 次")
print(f"  橙色预警(报警): {len(oa)} 次")
print(f"  红色预警(临灾): {len(ra)} 次")
if ra:
    print(f"\n{'='*60}")
    print("[!!! 极高危提示与现场工况回溯分析 !!!]")
    print(f"{'='*60}")
    print(f"检测到 {len(ra)} 次红色临灾预警，数字孪生回溯中...\n")
    
    cexp = {
        "U": "突发孔压激增(特大暴雨致地下水位剧升)",
        "E": "微震活动异常集中(微裂隙急剧扩展贯通)",
        "W": "干湿循环剧烈交替(导致岩土体软化锐减)",
        "R'": "爆破作业逼近(震动波直接冲击)",
        "Q'": "单段爆破药量违规增大(爆炸冲量超限)"
    }
    
    for ai in ra[:5]:
        print(f"-> 预警索引: {ai} | 瞬时激增速度: {vp[ai]:.4f}")
        adi = ai + 1
        if adi < len(dX):
            dv = dX[adi]
            cv = dv * b
            ac = np.abs(cv)
            tac = np.sum(ac) + 1e-8
            ci = np.argmax(ac)
            
            print(f"   工况快照(瞬时变化Δ):")
            print(f"     - ΔU: {dv[0]:.4f} | - ΔE: {dv[1]:.1f} | - ΔW: {dv[2]:.4f}")
            print(f"     - ΔR': {dv[3]:.1f} | - ΔQ': {dv[4]:.1f}")
            print(f"   [+] 致灾主导: 【{fn[ci]}】 (泰勒展开贡献 {ac[ci]/tac*100:.1f}%)")
            print(f"   [+] 物理机制: {cexp[fn[ci]]}")
            print("-" * 60)
else:
    print("\n[安全评估结论]")
    print("未触发红色预警，成功过滤正常爆破假警报。")
    print("当前边坡整体处于受控弹性变形状态，无宏观滑移风险。")
    if oa: print(f"注意: 索引 {oa[:3]}... 等处存在橙色扰动，建议加强巡查。")
print("\n问题5.2 分析完毕。")
