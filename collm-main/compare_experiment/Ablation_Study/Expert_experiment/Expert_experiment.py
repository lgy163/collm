import csv

import pandas as pd

from Agents.Simulation_Environment import SimulationEnvironment
from compare_experiment.Ablation_Study.Only_LLM.LLM import read_csv, get_weather_data, comfort_score_debug

# 加载数据
data = pd.read_csv()
FILE_PATH = ""
RESULT_PATH = ()


def combine_energy_comfort(energy: float, comfort: float, alpha: float = 0.5, beta: float = 0.5) -> float:
    # 归一化并反转能耗：当 energy = 250 时，score 为 1；当 energy = 730 时，score 为 0
    energy_score = (730.0 - energy) / 480.0  # 480 = 730 - 250
    energy_score = max(0.0, min(1.0, energy_score))
    # 舒适度直接使用，并确保在 [0,1] 范围内
    comfort_score = max(0.0, min(1.0, comfort))
    # 综合评价: 数值越大表示综合效果越好
    combined_val = alpha * energy_score + beta * comfort_score
    return combined_val


def apply_optimized_strategy(simulation_env, timestamp):
    """将优化后的策略值传递到仿真环境中"""
    T_out, Rh_out, precip, wind_speed, pressure = get_weather_data(
        '/home/lgy/Experiment/Code/Simulation_Model/台州椒江2024气象数据.xlsx',
        timestamp)
    P_chiller, T_cwr, T_cws, P_c_pump, P_tower, F_cw = simulation_env.get_result(simulation_env.CLc,
                                                                                 simulation_env.T_chws,
                                                                                 simulation_env.T_wet,
                                                                                 simulation_env.f_cwp,
                                                                                 simulation_env.tower_num,
                                                                                 simulation_env.f_chwp_list,
                                                                                 True)
    # 总能耗
    power_consumption = P_chiller + P_c_pump + P_tower

    c_comfort = comfort_score_debug(simulation_env.T_chwr, T_out, Rh_out, precip, wind_speed, pressure)

    return power_consumption, c_comfort

def main():
    model_path = "/home/lgy/Experiment/Code/Simulation_Model/save_model"
    ref_parameter = {
        'P_ref': 534.8,  # 冷机额定功率
        'COP_ref': 5.65,  # 冷机额定工况COP
        'C_ref': 534.8 * 5.65,  # 冷机额定制冷量
        'P_ch_pump_ref': 90,  # 冷却泵额定功率
        'Flow_cp_ref': 800,  # 冷却泵额定水流量
        'f_cp_ref': 50,  # 冷却泵额定频率
        'P_tower_ref': 11,  # 冷却塔额定功率
        'Flow_tower_ref': 1400,  # 冷却塔额定水流量
        'f_tower_ref': 50,  # 冷却塔额定频率
        'f_chp_ref': 50,  # 冷冻泵额定频率
        'Flow_chp_ref': 200,  # 冷冻泵额定水流量
    }

    simulation_env = SimulationEnvironment(ref_parameter=ref_parameter, model_path=model_path)

    with open(RESULT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "energy_consumption", "comfort_score", "metric"])

    for n, row in read_csv(FILE_PATH):
        timestamp = row['timestamp']
        simulation_env.read_environment_parameters(n, FILE_PATH)
        power_consumption, c_comfort = apply_optimized_strategy(simulation_env, timestamp)
        combined_val = combine_energy_comfort(power_consumption, c_comfort, alpha=0.5, beta=0.5)

        # 追加写入数据行
        with open(RESULT_PATH, 'a', newline='') as f:  # 注意用 'a' 追加模式
            writer = csv.writer(f)
            writer.writerow([timestamp, power_consumption, c_comfort, combined_val])


if __name__ == "__main__":
    main()
