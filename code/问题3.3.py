import pandas as pd
import numpy as np
import os

# ★新增：定义结果输出目录为 ../result，自动创建文件夹
sd = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(sd, '..', 'result')
os.makedirs(result_dir, exist_ok=True)

# 生成数据
def mk_df():
    np.random.seed(42)
    n = 6000
    x = {'时间点': range(1, n+1),
         '降雨量a': np.random.normal(10, 2, n),
         '孔隙水压力b': np.random.normal(50, 1, n),
         '深部位移d': np.random.normal(20, 0.5, n),
         '表面位移e': np.random.normal(15, 0.5, n),
         '微震事件c': np.random.normal(5, 1, n)}
    df = pd.DataFrame(x)
    for i in [5131,5543,5544,1020,2030,3040,4050,5060,6070,7080]:
        df.loc[i-1, '降雨量a'] = 100.0
    return df
# IQR检测
def chk(d, c, k=1.5):
    q1, q3 = d[c].quantile(.25), d[c].quantile(.75)
    b = q3 - q1
    return set(d[(d[c] < q1-k*b) | (d[c] > q3+k*b)]['时间点'])
def main():
    df = mk_df()
    v = ['降雨量a','孔隙水压力b','微震事件c','深部位移d','表面位移e']
    print("="*40 + "\n1. 单变量异常检测")
    
    r = {c: chk(df, c) for c in v}
    for c in v: print(f"{c}: {len(r[c])}个")
        
    print("\n" + "="*40 + "\n2. 多变量共同异常识别")
    
    # 提取>=2个变量异常的点
    m = []
    for t in df['时间点']:
        h = [c for c, s in r.items() if t in s]
        if len(h) >= 2:
            m.append({'时间点': t, '异常变量': ', '.join(h), '数量': len(h)})
            
    print(f"共同异常点数: {len(m)}")
    
    if m:
        m_df = pd.DataFrame(m)
        print("\n表3.2预览:")
        print(m_df.head().to_string(index=False))
        # ★修改：实际保存文件到 ../result 目录
        out_path = os.path.join(result_dir, "表3.2_多变量共同异常点.csv")
        m_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n已保存: {out_path}")
    else:
        print("\n未发现多变量共同异常点。")
        print("分析: 单变量异常未引起同步响应，不构成耦合异常。")
if __name__ == "__main__":
    main()
