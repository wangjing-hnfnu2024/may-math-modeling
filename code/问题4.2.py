import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestRegressor
import os
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# ★路径修改1：数据目录指向 ../data，结果目录指向 ../result 并自动创建
sd = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(sd, '..', 'data')
result_dir = os.path.join(sd, '..', 'result')
os.makedirs(result_dir, exist_ok=True)

# ★路径修改2：数据文件改为相对路径
fp = os.path.join(data_dir, "附件4：监测数据（训练集与实验集）-问题4.xlsx")

# 读数据
def ld(f, s):
    d = pd.read_excel(f, sheet_name=s)
    d.columns = [str(c).strip().replace('\n','').replace('\r','').replace(' ','') for c in d.columns]
    return d.dropna(axis=1, how='all').dropna(axis=0, how='all').reset_index(drop=True)
d_r = ld(fp, '训练集')
d_t = ld(fp, '实验集')
# 提特征
yc = '表面位移_mm'
tc = [c for c in d_r.columns if '时间' in c][0]
sc = next((c for c in d_t.columns if '阶段' in c or '标签' in c), None)
exc = [tc, yc] + ([sc] if sc else [])
fc = list(set([c for c in d_r.columns if c not in exc and d_r[c].dtype in ['float64','int64']]).union(
         set([c for c in d_t.columns if c not in exc and d_t[c].dtype in ['float64','int64']])))
Xr, yr = d_r[fc].fillna(0), d_r[yc]
Xt = d_t[fc].fillna(0)
# 训练预测
print("训练模型中...")
m = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
m.fit(Xr, yr)
d_t['预测表面位移_mm'] = m.predict(Xt)
d_t = d_t.sort_values(by=tc).reset_index(drop=True)
# 画图
print("绘制图像...")
f1, a1 = plt.subplots(figsize=(16, 7))
scp = ['#2196F3', '#FF5722', '#4CAF50']
ts = ["2025-05-09 12:00", "2025-05-27 08:00", "2025-06-01 12:00", "2025-06-03 22:00", "2025-06-04 01:40"]
if sc:
    sts = sorted(d_t[sc].dropna().unique())
    for i, st in enumerate(sts):
        sub = d_t[d_t[sc] == st]
        a1.plot(sub[tc], sub['预测表面位移_mm'], 
                color=scp[i%len(scp)], lw=1.8, label=f'阶段 {st}', alpha=0.9)
else:
    a1.plot(d_t[tc], d_t['预测表面位移_mm'], color='red', lw=1.5, label='预测位移')
a1.set_title('问题4.2：实验集表面位移预测时序曲线（分阶段）', fontsize=18, fontweight='bold')
a1.set_xlabel('时间', fontsize=14, fontweight='bold')
a1.set_ylabel('预测表面位移', fontsize=14, fontweight='bold')
a1.legend(loc='upper left', fontsize=12, framealpha=0.9)
a1.grid(True, alpha=0.3, ls='--')
f1.autofmt_xdate()
plt.tight_layout()
# 高亮特定点
for t_str in ts:
    tc_clean = t_str.replace("-","").replace(":","").replace(" ","")
    for idx, rt in enumerate(d_t[tc].astype(str)):
        if tc_clean in rt.replace("-","").replace(":","").replace(" ",""):
            a1.scatter(d_t.loc[idx, tc], d_t.loc[idx, '预测表面位移_mm'], 
                       color='red', s=50, zorder=5, edgecolors='black')
            break
# ★路径修改3：图片保存到 ../result
plt.savefig(os.path.join(result_dir, '图_4.2_表面位移预测曲线.png'), dpi=300, bbox_inches='tight')
plt.close()
print("图像已保存")
# 打印表格
print("\n" + "="*50)
print("         表4.1 补全结果提取")
print("="*50)
for t_str in ts:
    tc_clean = t_str.replace("-","").replace(":","").replace(" ","")
    idx = None
    for i, rt in enumerate(d_t[tc].astype(str)):
        rc = rt.replace("-","").replace(":","").replace(" ","")
        if tc_clean in rc or rc in tc_clean:
            idx = i
            break
            
    if idx is not None:
        print(f"时间: {t_str:<20} | 预测值: {d_t.loc[idx, '预测表面位移_mm']:.4f} mm")
    else:
        print(f"时间: {t_str:<20} | 警告：未找到该节点！")
print("="*50)
print("任务执行完毕！")
