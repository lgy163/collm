import csv
import os
import json

import numpy as np
import pandas as pd
from openai import OpenAI
from Agents.Simulation_Environment import SimulationEnvironment
import math

API_KEY = ""
BASE_URL = ""
FILE_PATH = ""
RESULT_PATH = ""

def read_csv(file_path):
    """按行读取CSV文件数据"""
    df = pd.read_csv(file_path)
    return df.iterrows()


def extract_content(response):
    """解析 API 响应并提取 content 字段的 JSON 数据"""
    response_dict = json.loads(response)  # 解析 JSON 字符串
    content = response_dict["choices"][0]["message"]["content"]  # 获取 content 字符串
    return json.loads(content)


def get_weather_data(file_path, timestamp):
    """根据给定的时间戳读取温度、湿度、降水、风速、气压的数据"""
    df = pd.read_excel(file_path, engine='openpyxl')  # 读取 Excel 文件
    df["时间"] = pd.to_datetime(df["时间"])  # 确保时间格式正确

    row = df[df["时间"] == pd.to_datetime(timestamp)]
    if row.empty:
        return None  # 若无匹配数据，则返回 None
    res = {
        "temperature": row["温度"].values[0],
        "humidity": row["湿度"].values[0],
        "precipitation": row["降水量"].values[0],
        "wind_speed": row["风速"].values[0],
        "pressure": row["气压"].values[0]
    }
    return res['temperature'], res['humidity'], res['precipitation'], res['wind_speed'], res['pressure']


def apply_optimized_strategy(simulation_env, timestamp, optimized_strategy):
    """将优化后的策略值传递到仿真环境中"""
    tower_num = optimized_strategy.get("tower_num", None)
    if tower_num is None:
        tower_num = optimized_strategy.get("Tower_num", None)
    simulation_env.set_control_parameters(
        T_chws=optimized_strategy["T_chws"],
        f_cwp=optimized_strategy["f_cwp"],
        tower_num=tower_num
    )
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


def call_init_model(api_key, base_url, env_info, strategy, tuning_range):
    """调用大模型API获取优化策略"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {'role': 'system',
         'content': '您是一个专业的冷水机组优化助手，在保持舒适的同时优化能源效率.'},
        {'role': 'user',
         'content': f'现有一套冷水机组系统，由4个冷却塔、1个冷却水泵、1个冷机、1个分水器、1个集水器组成，用于冷冻水供回水和冷却水供回水的循环。'
                    f'当前环境参数信息: {json.dumps(env_info, ensure_ascii=False)}'
                    f'当前策略: {json.dumps(strategy, ensure_ascii=False)}'
                    f'T_chws是冷冻水供水温度，调整范围是7.0到12.0每次可调的温度是0.5，f_cwp是冷却水泵的频率，可调整的范围是32.0-50.0，f_cwp最小步长0.2'
                    f'，tower_num是冷却塔的数量，可调整的范围是1到4tower_num最小步长1，'
                    f'请结合当前的T_chws,f_cwp,tower_num和系统能耗与舒适度做出判断调整T_chws,f_cwp,tower_num的值。'
                    f'请给出调控后的策略按照json格式输出，不要给出多余的理由，输出样例如下：'
                    f'{{“T_chws": <chilled water supply temperature>, '
                    f' “f_cwp": chilling pump frequency>,'
                    f' “tower_num": <number of cooling towers>.}}'}
    ]

    completion = client.chat.completions.create(
        model="qwen-max-latest",
        messages=messages
    )
    print(completion.model_dump_json())
    return extract_content(completion.model_dump_json())


def call_loop_model(api_key, base_url, power_consumption, c_comfort, strategy, tuning_range):
    """调用大模型API获取优化策略"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {'role': 'system',
         'content': '您是一个专业的冷水机组调优助手'},
        {'role': 'user',
         'content': f'通过上次的调优策略{json.dumps(strategy, ensure_ascii=False)}。'
                    f'---Strategy---通过你给出的参数调整方案，我们的出了新的系统能耗= {power_consumption}KW/h'
                    f'和舒适度 = {c_comfort}。'
                    f'请结合历史对话记录和知识库中的内容探索更优秀的策略'
                    f'输出格式仍然如下json格式，不需要给出多余信息。'
                    f'{{“T_chws": <chilled water supply temperature>, '
                    f' “f_cwp": chilling pump frequency>,'
                    f' “tower_num": <number of cooling towers>.}}'}
    ]

    completion = client.chat.completions.create(
        model="qwen-max-latest",
        messages=messages
    )
    return extract_content(completion.model_dump_json())


def dynamic_round_dialogue(env_info,
                           strategy,
                           tuning_range,
                           simulation_env,
                           timestamp,
                           max_rounds_initial=8,
                           max_rounds_limit=20,
                           improvement_threshold=0.001,
                           ):

    current_max_rounds = max_rounds_initial  # 当前允许的轮数上限
    total_rounds_done = 0  # 已完成轮数
    no_improve_count = 0  # 连续未改善计数

    # 用于保存最优结果
    best_metric = None
    best_info = None

    def combine_energy_comfort(energy: float, comfort: float, alpha: float = 0.5, beta: float = 0.5) -> float:

        # 归一化并反转能耗：当 energy = 250 时，score 为 1；当 energy = 730 时，score 为 0
        energy_score = (730.0 - energy) / 480.0  # 480 = 730 - 250
        energy_score = max(0.0, min(1.0, energy_score))
        # 舒适度直接使用，并确保在 [0,1] 范围内
        comfort_score = max(0.0, min(1.0, comfort))
        # 综合评价: 数值越大表示综合效果越好
        combined_val = alpha * energy_score + beta * comfort_score
        return combined_val

    while total_rounds_done < current_max_rounds:
        if total_rounds_done == 0:
            round_idx = total_rounds_done + 1
            response = call_init_model(API_KEY, BASE_URL, env_info, strategy, tuning_range)
            print("优化后的策略:", response)
            power_consumption, c_comfort = apply_optimized_strategy(simulation_env, timestamp, response)
            print(f"优化后的能耗:{power_consumption}，优化后的舒适度{c_comfort}")
        else:
            round_idx = total_rounds_done + 1
            strategy_current = call_loop_model(API_KEY, BASE_URL, power_consumption, c_comfort, strategy, tuning_range)
            print("优化后的策略:", strategy_current)
            power_consumption, c_comfort = apply_optimized_strategy(simulation_env, timestamp, strategy_current)
            print(f"优化后的能耗:{power_consumption}，优化后的舒适度{c_comfort}")
        # (2) 计算综合指标
        combined_val = combine_energy_comfort(power_consumption, c_comfort, alpha=0.5, beta=0.5)
        print(combined_val)

        # (3) 判断是否改善
        if best_metric is None:
            best_metric = combined_val
            best_info = (power_consumption, c_comfort, round_idx)
            no_improve_count = 0
        else:
            improvement = combined_val - best_metric
            if improvement > improvement_threshold:
                best_metric = combined_val
                best_info = (power_consumption, c_comfort, round_idx)
                no_improve_count = 0
            else:
                no_improve_count += 1

        total_rounds_done += 1  # 将计数放在外面，每次循环都累加

        if no_improve_count >= 5:
            print(f"已连续 {no_improve_count} 轮无明显改善, 提前终止.")
            break

        if total_rounds_done == current_max_rounds:
            if current_max_rounds < max_rounds_limit:
                new_limit = current_max_rounds + 5
                if new_limit > max_rounds_limit:
                    new_limit = max_rounds_limit
                print(f"到达 {current_max_rounds} 轮上限. 仍在改善, 扩容至 {new_limit} 轮.")
                current_max_rounds = new_limit
            else:
                print("已达最大对话轮数限制, 终止.")
                break
    print("\n对话结束!")
    print(f" - 共进行轮数: {total_rounds_done}")

    if best_info is not None:
        print(f" - 最优综合指标: {best_metric:.3f}")
        print(f" - 对应能耗: {best_info[0]:.2f}, 舒适度: {best_info[1]:.3f}, 轮次: {best_info[2]}")
    return best_metric, best_info


def main():
    model_path = "../../../Simulation_Model/save_model"
    max_rounds = 10

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

    # 第一步：判断 RESULT_PATH 是否存在，并统计已写入多少行
    start_line = 0  # 默认从第0行开始处理
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, 'r', newline='') as fr:
            reader = csv.reader(fr)
            existing_rows = list(reader)
            # 如果你的第一行是表头，则 existing_rows[0] 是表头
            # 数据从 existing_rows[1] 开始
            # 假设没有表头，则可直接用 len(existing_rows)
            # 或者若有表头，则有效数据行为 len(existing_rows) - 1
            if len(existing_rows) > 1:
                # 去掉表头后的数据行数
                data_lines = len(existing_rows) - 1
                start_line = data_lines  # 已写入了多少行，就跳过这些行
            else:
                # 只有表头 (或者甚至没有数据)
                start_line = 0
    else:
        # RESULT_PATH不存在，说明还没写入过任何数据
        start_line = 0
    print(start_line)
    if not os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "energy_consumption", "comfort_score", "metric"])
    for n, row in read_csv(FILE_PATH):
        if n < start_line:
            continue

        timestamp = row['timestamp']
        simulation_env.read_environment_parameters(n, FILE_PATH)
        env_info = {
            "Clc": row['CLc'],
            "T_wet": row['T_wet'],
            "T_chwr": row['T_chwr'],
            "T_chws": row['T_chws'],
            "f_cwp": row['f_cwp'],
            "Tower_num": row['Tower_num'],
            "f_tower": row['f_tower'],
        }
        strategy = {
            "T_chws": row['T_chws'],
            "f_cwp": row['f_cwp'],
            "Tower_num": row['Tower_num']
        }
        tuning_range = {
            "T_chws": [7.0, 12.0],
            "f_cwp": [32.0, 50.0],
            "Tower_num": [1, 4]
        }

        best_metric, best_info = dynamic_round_dialogue(env_info, strategy, tuning_range,
                                                        simulation_env, timestamp,
                                                        max_rounds_initial=10,
                                                        max_rounds_limit=20,
                                                        improvement_threshold=0.001,
                                                        )
        # 提取能耗、舒适度、指标
        energy = best_info[0]  # 能耗
        comfort = best_info[1]  # 舒适度
        metric = best_metric  # 指标

        # 追加写入数据行
        with open(RESULT_PATH, 'a', newline='') as f:  # 注意用 'a' 追加模式
            writer = csv.writer(f)
            writer.writerow([timestamp, energy, comfort, metric])


if __name__ == "__main__":
    main()

    # T_out, Rh_out, precip, wind_speed, pressure = get_weather_data(
    # '/home/lgy/Experiment/Code/Simulation_Model/台州椒江2024气象数据.xlsx', '2024/6/1 0:25') print(T_out, Rh_out, precip,
    # wind_speed, pressure)
