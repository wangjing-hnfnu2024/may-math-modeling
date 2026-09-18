import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# ★路径修改1：数据目录指向 ../data
sd = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(sd, '..', 'data')

# 底层指标计算
def ar2(y, yp, p):
    """计算调整R2"""
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    n = len(y)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
def mvif(X):
    """计算最大VIF"""
    vl = []
    for c in X.columns:
        y_a = X[c].values
        X_a = np.column_stack([np.ones(len(y_a)), X.drop(columns=[c]).values])
        m = LinearRegression(fit_intercept=False).fit(X_a, y_a)
        yp_a = m.predict(X_a)
        r2_a = 1 - np.sum((y_a - yp_a) ** 2) / np.sum((y_a - np.mean(y_a)) ** 2)
        vl.append(1 / (1 - r2_a) if r2_a != 1 else np.inf)
    return max(vl)
# 1. 数据读取与重构
# ★路径修改2：数据文件改为相对路径
fp = os.path.join(data_dir, "附件5：监测数据-问题5.xlsx")
print("读取数据...")
df = pd.read_excel(fp)
def gc(d, k):
    for c in d.columns:
        if k in str(c): return c
cy, cp = gc(df, '位移'), gc(df, '降雨')
cu = gc(df, '孔隙') or gc(df, '水压')
ce, cw = gc(df, '微震'), gc(df, '干湿') or gc(df, '入渗')
cr, cq = gc(df, '距离') or gc(df, '爆破点'), gc(df, '药量') or gc(df, '最大')
# 特征构建与滞后处理
dd = {
    'P': df[cp], 'U': df[cu], 'E': df[ce], 'W': df[cw],
    "R'": df[cr].fillna(0),  
    "Q'": df[cq].fillna(0)   
}
md = pd.DataFrame(dd)
md['U_lag1'] = md['U'].shift(1)
md['P_lag1'] = md['P'].shift(1)
md['Y'] = df[cy]
md = md.dropna()
Y = md['Y'].values
Xf = md[["P", "U", "E", "W", "R'", "Q'"]]
fn = Xf.columns.tolist()
N = len(Y)
print(f"有效样本 N={N}, 维度={len(fn)}\n")
# 2. 穷举与VIF双重筛选
ap = 1e5  
res = []
cbs = list(combinations(range(6), 5))
print("穷举与VIF诊断中...")
for cb in cbs:
    sf = [fn[i] for i in cb]
    X_sub = Xf[sf].values
    
    m = LinearRegression().fit(X_sub, Y)
    Yp = m.predict(X_sub)
    
    r2 = ar2(Y, Yp, 5)
    ee = 1 - r2
    ev = mvif(Xf[sf])
    
    pen = ap if ev > 10 else 0
    L = ee + pen
    dv = list(set(fn) - set(sf))[0]
    
    res.append({
        '保留变量组合': ", ".join(sf), '剔除变量': dv,
        'Adj_R2': r2, 'E_error': ee, 'Max_VIF': ev, 'L_score': L
    })
rdf = pd.DataFrame(res).sort_values(by='L_score').reset_index(drop=True)
print("="*60)
print("组合误差评估与共线性诊断结果表")
print("="*60)
print(rdf.to_string(index=False))
# 3. LASSO稀疏求解
print("\n" + "="*60)
print("基于 LASSO 的变量筛选结果")
print("="*60)
sc = StandardScaler()
Xs = sc.fit_transform(Xf)
lc = LassoCV(cv=5, random_state=42, max_iter=10000).fit(Xs, Y)
print(f"最优正则化系数 λ* = {lc.alpha_:.6f}")
print("各变量标准化LASSO系数：")
lcf = dict(zip(fn, lc.coef_))
zc = []
for f, c in lcf.items():
    st = "【剔除】" if np.isclose(c, 0, atol=1e-5) else f"保留 ({c:.4f})"
    print(f"  {f}: {st}")
    if np.isclose(c, 0, atol=1e-5): zc.append(f)
print("-" * 40)
if len(zc) == 1: print(f"结论: LASSO 恰好剔除 1 个变量 -> {zc[0]}")
elif len(zc) == 0: print("结论: LASSO 未剔除，以 OLS+VIF 机制决策为准。")
else: print(f"结论: LASSO 剔除 {len(zc)} 个，以 OLS+VIF 机制决策为准。")
# 4. 最终结论输出
print("\n" + "="*60)
print("【问题5.1 最终评估结论】")
print("="*60)
best = rdf.iloc[0]
print(f"最优5类变量组合: {best['保留变量组合']}")
print(f"被剔除的变量:   {best['剔除变量']}")
print(f"调整决定系数: Adj_R2 = {best['Adj_R2']:.4f} (解释误差 = {best['E_error']:.4f})")
vif_st = '(<=10，无共线性)' if best['Max_VIF'] <= 10 else '(>10，被惩罚淘汰)'
print(f"最大膨胀因子: Max_VIF = {best['Max_VIF']:.4f} {vif_st}")
print("\n【原理机制解释】")
print("1. 物理重构：将空爆破数据赋0重构R'和Q'，维持6维候选池，规避物理谬误引发的扰动。")
print("2. 误差泛函：采用Adj_R2转化E_error，剔除虚假拟合上升，真实反映定量解释误差。")
print("3. 双重筛选：采用LASSO做L1稀疏降维探路；引入VIF作共线性惩罚，确保最终决策在最高精度与无多重共线性间取得最优平衡。")
