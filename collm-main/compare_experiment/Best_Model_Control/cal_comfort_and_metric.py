import math
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from Agents.Simulation_Environment import SimulationEnvironment

# 配置参数
model_path = "/home/lgy/Experiment/Code/Simulation_Model/save_model"
ref_parameter = {
    'P_ref': 534.8,
    'COP_ref': 5.65,
    'C_ref': 534.8 * 5.65,
    'P_ch_pump_ref': 90,
    'Flow_cp_ref': 800,
    'f_cp_ref': 50,
    'P_tower_ref': 11,
    'Flow_tower_ref': 1400,
    'f_tower_ref': 50,
    'f_chp_ref': 50,
    'Flow_chp_ref': 200,
}

timestamp_file_path = "/home/lgy/Experiment/Code/Execute_Model/july.csv"
weather_file_path   = '/home/lgy/Experiment/Code/Simulation_Model/台州椒江2024气象数据.xlsx'
output_file_path    = "/home/lgy/Experiment/Code/compare_experiment/Best_Model_Control/july_result_with_0.55comfort.csv"

# 离散参数空间
t_chws_values    = np.round(np.arange(9.0, 12.01, 1), 1)
tower_num_values = [1, 2, 3, 4]
f_cwp_values     = np.round(np.arange(32.0, 50.01, 1), 1)
param_combinations = list(product(t_chws_values, tower_num_values, f_cwp_values))

def get_weather_data(file_path, timestamp):
    """根据时间戳获取天气数据"""
    df = pd.read_excel(file_path, engine='openpyxl')
    df["时间"] = pd.to_datetime(df["时间"])
    row = df[df["时间"] == pd.to_datetime(timestamp)]
    if row.empty:
        return None
    return (
        row["温度"].values[0], row["湿度"].values[0],
        row["降水量"].values[0], row["风速"].values[0],
        row["气压"].values[0]
    )

def comfort_score_debug(T_chwr, T_out, RH_out, precip, wind_speed, pressure):
    """计算舒适度评分"""
    # 室内指数衰减
    T_opt, alpha = 12.0, 0.03
    indoor = math.exp(-alpha * (T_chwr - T_opt)**2)
    # 室外五维评分
    def score_wind(w):
        w_opt = 2.0
        γ1, γ2 = 0.05, 0.3
        return math.exp(-γ1*(w-w_opt)**2) if w <= w_opt else math.exp(-γ2*(w-w_opt)**2)
    scores = {
        'temp': math.exp(-0.02*(T_out - 25)**2),
        'hum':  math.exp(-0.015*(RH_out - 50)**2),
        'precip': math.exp(-0.1*precip),
        'wind': score_wind(wind_speed),
        'pressure': math.exp(-0.005*(pressure - 1013)**2)
    }
    weights = {'temp':0.3, 'hum':0.2, 'precip':0.25, 'wind':0.15, 'pressure':0.1}
    f_outdoor = np.prod([scores[k]**weights[k] for k in scores])
    # 动态权重
    p_w = 1/(1+math.exp(-0.5*(precip-5)))
    w_out, w_in = min(0.6, 0.3+0.3*p_w), None
    w_in = 1 - w_out
    return w_in*indoor + w_out*f_outdoor

def combine_energy_comfort(energy, comfort, alpha=0.45, beta=0.55):
    e_score = (730.0 - energy)/480.0
    e_score = max(0, min(1, e_score))
    c_score = max(0, min(1, comfort))
    return alpha*e_score + beta*c_score

def process_timestamp(n):
    """处理单个时间戳的核心函数"""
    env = SimulationEnvironment(ref_parameter=ref_parameter, model_path=model_path)
    env.read_environment_parameters(n, timestamp_file_path)
    df_ts = pd.read_csv(timestamp_file_path)
    timestamp = df_ts.iloc[n]["timestamp"]
    weather = get_weather_data(weather_file_path, timestamp)
    if not weather:
        return None
    T_out, RH_out, precip, wind_speed, pressure = weather

    best = {'combined': -np.inf, 'params':None, 'energy':None, 'comfort':None, 'cop':None}
    for t_chws, tower_num, f_cwp in param_combinations:
        env.T_chws     = t_chws
        env.tower_num  = tower_num
        env.f_cwp      = f_cwp
        try:
            P_chiller, _, _, P_c_pump, P_tower, _ = env.get_result(
                env.CLc, env.T_chws, env.T_wet, env.f_cwp,
                env.tower_num, env.f_chwp_list, True
            )
        except:
            continue
        power = P_chiller + P_c_pump + P_tower
        cop   = env.CLc/power if power>0 else 0

        comfort = comfort_score_debug(env.T_chwr, T_out, RH_out, precip, wind_speed, pressure)
        combined = combine_energy_comfort(power, comfort)
        if combined > best['combined']:
            best.update({
                'combined': combined,
                'params':  (t_chws, tower_num, f_cwp),
                'energy':  power,
                'comfort': comfort,
                'cop':     cop
            })
    print(best['energy'], best['comfort'], best['combined'], best['cop'])
    return (timestamp, *best['params'], best['energy'], best['comfort'], best['combined'], best['cop'])

def main():
    df = pd.read_csv(timestamp_file_path)
    total = len(df)
    results = []
    for n in tqdm(range(total), desc="Processing"):
        res = process_timestamp(n)
        if res:
            results.append(res)

    output_df = pd.DataFrame(results, columns=[
        'timestamp', 'T_chws', 'tower_num', 'f_cwp',
        'energy', 'comfort', 'combined', 'cop'
    ])
    output_df.to_csv(output_file_path, index=False)
    print("Processing completed. Results saved to:", output_file_path)

if __name__ == "__main__":
    main()
