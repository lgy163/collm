import pandas as pd
import os

# ---------- 路径 ----------
expert_raw   = r'D:\pyproject\pythonProject\Simulation_Model\data.csv'
expert_june  = r'D:\pyproject\pythonProject\compare_experiment\Ablation_Study\Expert_experiment\Expert_Experiment_june_result.csv'
save_path    = r'D:\pyproject\pythonProject\Simulation_Model\expert_june_energy_comfort.csv'

for p in (expert_raw, expert_june):
    if not os.path.exists(p):
        raise FileNotFoundError(p)

# ---------- 1. 读取原始数据并选取 6 月 ----------
df_raw = pd.read_csv(expert_raw, parse_dates=['timestamp'])

# 仅保留时间戳在 6 月 (month == 6) 的行
df_raw_june = df_raw[df_raw['timestamp'].dt.month == 6].copy()

# ---------- 2. 计算总能耗 ----------
df_raw_june['energy_consumption'] = (
    df_raw_june['P_chiller'] +
    df_raw_june['P_tower']   +
    df_raw_june['P_cwp']
)

# 只保留需要列并重命名 COP_s 为 cop
df_raw_june = df_raw_june[['timestamp', 'energy_consumption', 'COP_s']].rename(
    columns={'COP_s': 'cop'}
)

# ---------- 3. 读取 comfort_score ----------
df_comf = pd.read_csv(expert_june, parse_dates=['timestamp'])
df_comf = df_comf[['timestamp', 'comfort_score']]

# ---------- 4. 合并 ----------
result = (
    df_raw_june
    .merge(df_comf, on='timestamp', how='inner')   # 按时间戳对齐
    .sort_values('timestamp')
    .reset_index(drop=True)
)

# ---------- 5. 保存 ----------
os.makedirs(os.path.dirname(save_path), exist_ok=True)
result.to_csv(save_path, index=False)
print(f'新文件已保存至: {save_path}')
