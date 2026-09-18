import pandas as pd
import numpy as np
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# ★路径修改1：数据目录指向 ../data
sd = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(sd, '..', 'data')
result_dir = os.path.join(sd, '..', 'result')
os.makedirs(result_dir, exist_ok=True)

# 智能去噪与缺失值补齐
def clean(f):
    print(f"读取: {f}")
    df = pd.read_excel(f)
    
    # 提取数值列并剔除序号列
    cols = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if not df[c].equals(pd.Series(range(1, len(df)+1))):
            cols.append(c)
    df = df[cols].copy()
    
    # 清理表头换行符
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    print(f"提取 {len(df.columns)} 个特征")
    
    # 线性插值补齐缺失值
    if df.isnull().sum().sum() > 0:
        print("缺失值插值中...")
        df = df.interpolate(method='linear', limit_direction='both').bfill().ffill()
        
    # 中值滤波与IQR去噪
    print("去噪处理中...")
    df_out = df.copy()
    for c in df_out.columns:
        s = df_out[c]
        
        # 滑动中值滤波
        w = 3 if len(s) < 50 else 5
        med = s.rolling(window=w, center=True, min_periods=1).median()
        
        # IQR异常值检测与替换
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        out = (s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)
        
        if out.sum() > 0:
            print(f"  {c[:10]}: 替换 {out.sum()} 个异常值")
            df_out[c] = np.where(out, med, s)
        else:
            df_out[c] = med
            
    print("去噪完成\n")
    return df_out

if __name__ == "__main__":
    # ★路径修改2：搜索 data 目录下的"附件3"文件（原代码搜英文Attachment+当前目录，改中文+data目录）
    files = glob.glob(os.path.join(data_dir, "*附件3*.xlsx"))
    if not files:
        # ★路径修改3：给出更明确的未找到提示
        print(f"未找到文件：请在 data 目录确认是否有「附件3」开头的xlsx文件")
        print(f"当前搜索目录: {data_dir}")
        print(f"data目录现有文件: {os.listdir(data_dir) if os.path.isdir(data_dir) else '目录不存在'}")
    else:
        path = files[0]
        print(f"定位文件: {path}\n" + "-"*40)
        
        # 清洗并保存
        res = clean(path)
        res.insert(0, 'No.', range(1, len(res) + 1))
        
        # ★路径修改4：输出到 ../result 文件夹
        out_path = os.path.join(result_dir, "清洗后数据.xlsx")
        res.to_excel(out_path, index=False)
        
        print("-"*40 + f"\n已保存至: {out_path}")
