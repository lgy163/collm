import pandas as pd
import math
import numpy as np

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
        'precip': math.exp(-0.1 * precip),  # 指数衰减
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


def main():
    # 1) 读取 output.csv 文件，包括 timestamp、T_chwr 等列
    df_output = pd.read_csv('output.csv')

    # 2) 读取环境数据，如 env_data.xlsx。若是 CSV 则用 pd.read_csv
    # 这里假设 xlsx 文件中有列: timestamp, T_out, RH_out, precip, wind_speed, pressure
    df_env = pd.read_excel('C:\\Users\\Administrator\\Desktop\\hvac知识文档\\台州中心医院\\台州椒江2024气象数据.xlsx')

    # 3) 如果需要按 timestamp 合并, 确保两者 timestamp 格式一致
    # 例如，把 timestamp 转为 datetime 或字符串统一:
    # df_output['timestamp'] = pd.to_datetime(df_output['timestamp'])
    # df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])

    # 4) 通过 timestamp 字段合并，将 env_data 列并入 df_output
    df_merged = pd.merge(df_output, df_env, on='timestamp', how='left')

    # df_merged 现包含: [timestamp, T_chwr, ... 以及 T_out, RH_out, precip, wind_speed, pressure]

    # 5) 逐行计算 comfort_score
    comfort_scores = []
    for idx, row in df_merged.iterrows():
        T_chwr_val   = row['T_chwr']
        T_out_val    = row['T_out']
        RH_out_val   = row['RH_out']
        precip_val   = row['precip']
        wind_val     = row['wind_speed']
        pressure_val = row['pressure']

        c_score = comfort_score_debug(
            T_chwr=T_chwr_val,
            T_out=T_out_val,
            RH_out=RH_out_val,
            precip=precip_val,
            wind_speed=wind_val,
            pressure=pressure_val
        )
        comfort_scores.append(c_score)

    # 6) 将计算好的舒适度写回 df_merged 的一个新列 comfort_score
    df_merged['comfort_score'] = comfort_scores

    # 7) 将合并和计算完的 df_merged 根据需要保存到 output.csv 或新文件
    # 如果想继续覆盖 output.csv 的内容(包括新列):
    df_merged.to_csv('output.csv', index=False)

    print("Calculation done, output saved to output.csv")


if __name__ == "__main__":
    main()
