import pandas as pd
import numpy as np
import os

# ★新增：结果保存目录指向 ../result 并自动创建
sd = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(sd, '..', 'result')
os.makedirs(result_dir, exist_ok=True)

# 生成模拟数据
def gen_data():
    np.random.seed(42)
    n = 6000
    d = {
        '时间点': np.arange(1, n + 1),
        '降雨量a': np.random.normal(10, 2, n),
        '孔隙水压力b': np.random.normal(50, 1, n),
        '深部位移d': np.random.normal(20, 0.5, n),
        '表面位移e': np.random.normal(15, 0.5, n),
        '微震事件c': np.random.normal(5, 1, n)
    }
    df = pd.DataFrame(d)
    
    # 注入异常点
    for i in [5131, 5543, 5544, 1020, 2030, 3040, 4050, 5060, 6070, 7080]:
        df.loc[i - 1, '降雨量a'] = 100.0
    return df

# IQR异常检测
def get_out(df, col, k=1.5):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - k * IQR, Q3 + k * IQR
    return set(df[(df[col] < lo) | (df[col] > hi)]['时间点'].tolist())

def run():
    df = gen_data()
    cols = ['降雨量a', '孔隙水压力b', '微震事件c', '深部位移d', '表面位移e']
    
    print("="*40)
    print("1. 单变量异常检测")
    
    res = {}
    for c in cols:
        res[c] = get_out(df, c)
        print(f"{c}: {len(res[c])}个")
        
    print("\n" + "="*40)
    print("2. 多变量共同异常识别")
    
    # 筛选同时>=2个变量异常的时间点
    com = []
    for t in df['时间点']:
        hit = [c for c, s in res.items() if t in s]
        if len(hit) >= 2:
            com.append({'时间点': t, '异常变量': ', '.join(hit), '数量': len(hit)})
            
    print(f"共同异常点数: {len(com)}")
    
    if com:
        com_df = pd.DataFrame(com)
        print("\n表3.2预览:")
        print(com_df.head().to_string(index=False))
        
        # ★真正保存CSV到 result 文件夹
        out_path = os.path.join(result_dir, '表3.2_多变量共同异常点.csv')
        com_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n已保存: {out_path}")
    else:
        print("\n未发现多变量共同异常点。")
        print("分析: 降雨量a单变量异常未引起其他变量同步响应，不构成耦合异常。")

if __name__ == "__main__":
    run()
