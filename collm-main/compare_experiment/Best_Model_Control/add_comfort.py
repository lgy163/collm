import pandas as pd
import numpy as np
import math

# 读取 result_control_1.csv 文件
df_control = pd.read_csv(r'best_model_result.csv')

# 读取气象数据文件
weather_df = pd.read_excel(r'D:\pyproject\pythonProject\Simulation_Model\台州椒江2024气象数据.xlsx')


# 定义舒适度评分函数
def comfort_score_debug(
        T_chwr: float,
        T_out: float,
        RH_out: float,
        precip: float,
        wind_speed: float,
        pressure: float,
) -> float:
    ################## 室内评分 (举例指数衰减法) ##################
    T_opt_chwr = 12.0
    alpha_chwr = 0.03
    dist_chwr = (T_chwr - T_opt_chwr) ** 2
    indoor_score = math.exp(- alpha_chwr * dist_chwr)

    # --------------- 室外评分 ---------------#
    # 非对称风速评分
    def score_wind(w):
        w_opt = 2.0
        if w <= w_opt:
            return math.exp(-0.05 * (w - w_opt) ** 2)
        else:
            return math.exp(-0.3 * (w - w_opt) ** 2)

    # 动态权重
    outdoor_weights = {
        'temp': 0.3,
        'hum': 0.2,
        'precip': 0.25,
        'wind': 0.15,
        'pressure': 0.1
    }

    scores = {
        'temp': math.exp(-0.02 * (T_out - 25) ** 2),
        'hum': math.exp(-0.015 * (RH_out - 50) ** 2),
        'precip': math.exp(-0.1 * precip),  # 指数衰减更合理
        'wind': score_wind(wind_speed),
        'pressure': math.exp(-0.005 * (pressure - 1013) ** 2)
    }

    # 加权几何平均
    f_outdoor = np.prod([s ** w for s, w in zip(scores.values(), outdoor_weights.values())])

    # 动态权重分配
    precip_weight = 1 / (1 + math.exp(-0.5 * (precip - 5)))
    w_outdoor = min(0.6, 0.3 + 0.3 * precip_weight)  # 雨天提升室外权重
    w_indoor = 1 - w_outdoor
    c_comfort = w_indoor * indoor_score + w_outdoor * f_outdoor
    return c_comfort



# 创建一个空列表来存储计算后的舒适度得分
comfort_scores = []

# 遍历 result_control_1.csv 中的每一行数据
for index, row in df_control.iterrows():
    # 提取当前行的 T_chwr 值
    T_chwr = row['T_chwr']

    # 获取对应时间戳的气象数据（假设时间戳在两个数据集中都存在）
    timestamp = row['timestamp']
    weather_data = weather_df[weather_df['时间'] == timestamp].iloc[0]

    # 提取气象数据（温度、湿度、降水量、风速、气压）
    T_out = weather_data['温度']  # 室外温度
    RH_out = weather_data['湿度']  # 室外湿度
    precip = weather_data['降水量']  # 降水量
    wind_speed = weather_data['风速']  # 风速
    pressure = weather_data['气压']  # 气压

    # 计算舒适度得分
    comfort = comfort_score_debug(T_chwr, T_out, RH_out, precip, wind_speed, pressure)

    # 将计算得到的舒适度得分添加到列表中
    comfort_scores.append(comfort)

# 将舒适度得分添加到 DataFrame 中
df_control['comfort'] = comfort_scores

# 计算 P_total 列并添加到 DataFrame
df_control['P_total'] = df_control['P_chiller'] + df_control['P_cwp'] + df_control['P_tower']

# 保存修改后的 DataFrame 到新的 CSV 文件
df_control.to_csv(r'best_model_updated.csv', index=False)

print("计算完成，已保存结果至 best_model_updated.csv")
