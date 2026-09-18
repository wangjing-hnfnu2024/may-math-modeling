import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pandas.plotting import scatter_matrix
import warnings
from scipy.stats import gaussian_kde
import os
warnings.filterwarnings('ignore')
plt.ioff()
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# ★路径修改1：数据目录指向 ../data
sd = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(sd, '..', 'data')
result_dir = os.path.join(sd, '..', 'result')
os.makedirs(result_dir, exist_ok=True)

# ★路径修改2：数据文件路径改为相对路径（指向data目录）
fp = os.path.join(data_dir, "附件4：监测数据（训练集与实验集）-问题4.xlsx")

# 读取与清洗
def ld(f, s):
    d = pd.read_excel(f, sheet_name=s)
    d.columns = [str(c).strip().replace('\n','').replace('\r','').replace(' ','') for c in d.columns]
    return d.dropna(axis=1, how='all').dropna(axis=0, how='all').reset_index(drop=True)
d_r = ld(fp, '训练集')
d_t = ld(fp, '实验集')
# 智能识别列
mc, mp = [], {}
for c in d_t.columns:
    if '时间' in c: mp['t'] = c
    elif '阶段' in c or '标签' in c: mp['s'] = c
    elif any(x in c for x in ['表面位移','内部位移','锚杆','轴力','渗透','水压','孔隙','温度']): 
        mc.append(c)
tc, sc = mp.get('t', d_t.columns[0]), mp.get('s', None)
sts = sorted(d_t[sc].dropna().unique()) if sc else None
cp = ['#2196F3','#FF5722','#4CAF50','#FF9800','#9C27B0','#00BCD4','#E91E63','#795548']
scd = {sts[i]: cp[i%len(cp)] for i in range(len(sts))} if sts else {}
# 辅助函数：重命名与降采样
def s1(c): return c.replace('表面位移_mm','表面位移').replace('内部位移_mm','内部位移').replace('锚杆轴力_kN','锚杆轴力').replace('渗透压力_kPa','渗透压力').replace('孔隙水压力_kPa','孔隙水压力').replace('温度_℃','温度(℃)')
def s2(c): return s1(c).replace('温度(℃)', '温度')
def ds(d, n=500): return d if len(d) <= n else d.iloc[::len(d)//n].copy()
print("生成图片...")
# ======================== 图1：实验集时序 ========================
f1, a1 = plt.subplots(len(mc), 1, figsize=(14, 3.2*len(mc)))
if len(mc) == 1: a1 = [a1]
f1.suptitle('实验集监测数据时序曲线', fontsize=18, fontweight='bold', y=0.98)
for i, c in enumerate(mc):
    ax = a1[i]
    if sts:
        for st in sts:
            sub = ds(d_t[d_t[sc] == st], 400)
            if len(sub) > 0:
                ax.plot(sub[tc], sub[c], color=scd[st], lw=1.2, label='阶段'+str(st), alpha=0.9)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    mx, mn = d_t[c].idxmax(), d_t[c].idxmin()
    ax.annotate('最大: '+str(round(d_t[c].max(),2)), 
                xy=(d_t[tc].iloc[mx], d_t[c].max()), fontsize=8, color='red', fontweight='bold', 
                xytext=(15, 10), textcoords='offset points', 
                arrowprops=dict(arrowstyle='->', color='red', lw=1), 
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='none'))
    ax.annotate('最小: '+str(round(d_t[c].min(),2)), 
                xy=(d_t[tc].iloc[mn], d_t[c].min()), fontsize=8, color='blue', fontweight='bold', 
                xytext=(15, -20), textcoords='offset points', 
                arrowprops=dict(arrowstyle='->', color='blue', lw=1), 
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='none'))
                
    ax.set_ylabel(s1(c), fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, ls='--')
    ax.tick_params(labelsize=9)
a1[-1].set_xlabel('时间', fontsize=12, fontweight='bold')
f1.autofmt_xdate()
f1.tight_layout(rect=[0, 0, 1, 0.96])
# ★路径修改3：图片保存到 ../result
plt.savefig(os.path.join(result_dir, '图1_实验集监测时序曲线.png'), dpi=200, bbox_inches='tight')
plt.close(f1)
print("图1已保存")
# ======================== 图2：训练集时序 ========================
f2, a2 = plt.subplots(len(mc), 1, figsize=(14, 3.2*len(mc)))
if len(mc) == 1: a2 = [a2]
f2.suptitle('训练集监测数据时序曲线', fontsize=18, fontweight='bold', y=0.98)
for i, c in enumerate(mc):
    ax = a2[i]
    sub = ds(d_r, 500)
    ax.plot(sub[tc], sub[c], color='#4CAF50', lw=1.0, alpha=0.85)
    ax.set_ylabel(s1(c), fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, ls='--')
    ax.tick_params(labelsize=9)
a2[-1].set_xlabel('时间', fontsize=12, fontweight='bold')
f2.autofmt_xdate()
f2.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(result_dir, '图2_训练集监测时序曲线.png'), dpi=200, bbox_inches='tight')
plt.close(f2)
print("图2已保存")
# ======================== 图3：训练 vs 实验 ========================
f3, a3 = plt.subplots(len(mc), 1, figsize=(14, 3.2*len(mc)))
if len(mc) == 1: a3 = [a3]
f3.suptitle('训练集 vs 实验集 监测数据对比', fontsize=18, fontweight='bold', y=0.98)
for i, c in enumerate(mc):
    ax = a3[i]
    t1, t2 = ds(d_r, 400), ds(d_t, 400)
    ax.plot(t1[tc], t1[c], color='#2196F3', lw=1.0, alpha=0.7, label='训练集')
    ax.plot(t2[tc], t2[c], color='#FF5722', lw=1.0, alpha=0.7, label='实验集')
    ax.set_ylabel(s1(c), fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls='--')
    ax.tick_params(labelsize=9)
a3[-1].set_xlabel('时间', fontsize=12, fontweight='bold')
f3.autofmt_xdate()
f3.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(result_dir, '图3_训练实验集对比.png'), dpi=200, bbox_inches='tight')
plt.close(f3)
print("图3已保存")
# ======================== 图4：箱线图 ========================
f4, a4 = plt.subplots(2, 3, figsize=(16, 10))
f4.suptitle('各监测指标分布箱线图', fontsize=18, fontweight='bold', y=0.98)
a4 = a4.flatten()
for i, c in enumerate(mc):
    ax, bd, bl = a4[i], [], []
    if sts:
        for st in sts:
            d = d_t.loc[d_t[sc] == st, c].dropna()
            if len(d) > 0: 
                bd.append(d.values)
                bl.append(str(st))
        if bd:
            bp = ax.boxplot(bd, patch_artist=True, medianprops=dict(color='red', lw=2))
            ax.set_xticklabels(bl)
            for j, p in enumerate(bp['boxes']): 
                p.set_facecolor(scd.get(bl[j], '#2196F3'))
                p.set_alpha(0.6)
    ax.set_title(s2(c), fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', ls='--')
for i in range(len(mc), len(a4)): a4[i].set_visible(False)
f4.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(result_dir, '图4_箱线图分布.png'), dpi=200, bbox_inches='tight')
plt.close(f4)
print("图4已保存")
# ======================== 图5：热力图 ========================
nc = [c for c in mc if d_t[c].dtype in ['float64','int64','float32','int32']]
cr = d_t[nc].corr()
cl = ['#2166AC','#67A9CF','#D1E5F0','#F7F7F7','#FDDBC7','#EF8A62','#B2182B']
cmap = LinearSegmentedColormap.from_list('c', cl, N=256)
f5, ax5 = plt.subplots(figsize=(10, 8))
im = ax5.imshow(cr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
lb = [s2(c) for c in cr.columns]
ax5.set_xticks(range(len(lb)))
ax5.set_yticks(range(len(lb)))
ax5.set_xticklabels(lb, fontsize=11, rotation=30, ha='right')
ax5.set_yticklabels(lb, fontsize=11)
for i in range(len(cr)):
    for j in range(len(cr)):
        v = cr.values[i, j]
        ax5.text(j, i, str(round(v,3)), ha='center', va='center', 
                 fontsize=11, fontweight='bold', color='white' if abs(v)>0.6 else 'black')
plt.colorbar(im, ax=ax5, shrink=0.8).set_label('相关系数', fontsize=12)
ax5.set_title('监测指标相关性热力图', fontsize=16, fontweight='bold', pad=15)
f5.tight_layout()
plt.savefig(os.path.join(result_dir, '图5_相关性热力图.png'), dpi=200, bbox_inches='tight')
plt.close(f5)
print("图5已保存")
# ======================== 图6：散点矩阵 ========================
pc = nc[:4]
dp = ds(d_t[pc], 500).copy()
dp.columns = [s2(c) for c in pc]
n = len(pc)
xaa = scatter_matrix(dp, figsize=(14, 14), alpha=0.5, diagonal='hist', 
                     hist_kwds=dict(bins=25, edgecolor='white'), color='#2196F3')
xaa = np.array(xaa).reshape(n, n)
for i in range(n):
    ax = xaa[i, i]
    ax.clear()
    val = dp.iloc[:, i].dropna().values
    
    if len(val) > 1 and np.std(val) > 0:
        kde = gaussian_kde(val, bw_method=0.3)
        x_e = np.linspace(val.min(), val.max(), 200)
        y_e = kde(x_e)
        ax.plot(x_e, y_e, color='#2196F3', lw=1.5)
        ax.fill_between(x_e, y_e, color='#2196F3', alpha=0.3)
    elif len(val) > 0: 
        ax.axvline(val[0], color='#2196F3', lw=1.5)
        
    ax.set_title(dp.columns[i], fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, ls='--')
f6 = xaa[0, 0].figure
f6.suptitle('监测指标散点矩阵图', fontsize=18, fontweight='bold', y=1.02)
plt.savefig(os.path.join(result_dir, '图6_散点矩阵图.png'), dpi=200, bbox_inches='tight')
plt.close(f6)
print("图6已保存")
print("\n全部图片已保存!")
