import pandas as pd

from Agents.Simulation_Environment import SimulationEnvironment

if __name__ == "__main__":
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
    model_path = "../../Simulation_Model/save_model"  # 模型文件路径
    simulation = SimulationEnvironment(ref_parameter, model_path)

    # 读取CSV文件
    file_path = r'best_model_result.csv'
    df = pd.read_csv(file_path)

    # 创建空的T_chwr列
    df['T_chwr'] = None

    # 对每一行计算T_chwr并添加新列
    for index, row in df.iterrows():
        simulation.read_environment_parameters_for_fine_tuning_LLM(index, file_path)
        f_chwp1, f_chwp2, f_chwp3,f_chwp4, f_chwp5, f_chwp6 = simulation.f_chwp_list
        F_carrier = simulation.get_F_carrier(simulation.CLc, f_chwp1, f_chwp2, f_chwp3,
                                             f_chwp4, f_chwp5, f_chwp6)
        T_chwr = simulation.get_T_chwr(simulation.T_chws, simulation.CLc, F_carrier)
        df.at[index, 'T_chwr'] = T_chwr

    # 保存结果到新的CSV文件
    df.to_csv('best_model_result_with_T_chwr.csv', index=False)