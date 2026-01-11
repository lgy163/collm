import os, re, csv
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status
import math
import json
import pandas as pd
from Agents.Simulation_Environment import SimulationEnvironment

WORKING_DIR = "./dickens"
API_KEY = ""
BASE_URL = ""
FILE_PATH = ""
RESULT_PATH = ""

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=True, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="text-embedding-v3",
        api_key=API_KEY,
        base_url=BASE_URL,
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())


async def initialize_rag():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_cache_config={
            "enabled": False,
            "similarity_threshold": 0.90,
        },
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
        entity_extract_max_gleaning=10,
        addon_params={
            "insert_batch_size": 5  # 每批处理20个文档
        }
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def read_csv(file_path):
    """按行读取CSV文件数据"""
    df = pd.read_csv(file_path)
    return df.iterrows()


def extract_content(response):
    """解析 API 响应并提取 content 字段的 JSON 数据"""
    response_dict = json.loads(response)  # 解析 JSON 字符串
    content = response_dict["choices"][0]["message"]["content"]  # 获取 content 字符串
    return json.loads(content)


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
        '',
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


def update_strategy_from_response(response: str, current_strategy: dict) -> dict:
    """从响应文本中提取JSON参数并更新策略，支持带或不带```json```包裹的JSON格式"""
    # 步骤1：使用正则表达式尝试提取带 ```json ``` 包裹的 JSON
    fenced_pattern = r'```json\s*({.*?})\s*```'

    # 回退匹配任意裸 JSON（支持多行、多键）
    raw_pattern = r'(\{[\s\S]*?\})'
    match = re.search(fenced_pattern, response, re.DOTALL)

    # 若未找到，则尝试提取裸 JSON 块
    if not match:
        match = re.search(raw_pattern, response, re.DOTALL)

    if not match:
        raise ValueError("未在响应中找到有效的JSON参数块")

    # 步骤2：解析JSON数据
    try:
        optimized_params = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {str(e)}") from e

    # 步骤3：键名标准化处理
    param_mapping = {
        "tower_num": "Tower_num",
        "Tower_num": "Tower_num",
        "T_chws": "T_chws",
        "f_cwp": "f_cwp"
    }

    # 步骤4：更新策略参数
    updated_strategy = current_strategy.copy()
    for resp_key, strategy_key in param_mapping.items():
        if resp_key in optimized_params:
            original_value = current_strategy.get(strategy_key)
            new_value = optimized_params[resp_key]
            # 保持原有数据类型（如 int/float）
            if original_value is not None:
                updated_strategy[strategy_key] = type(original_value)(new_value)
            else:
                updated_strategy[strategy_key] = new_value

    # 步骤5：验证必要参数是否完整
    required_keys = ["T_chws", "f_cwp", "Tower_num"]
    for key in required_keys:
        if key not in updated_strategy:
            raise ValueError(f"缺失必要参数: {key}")

    return updated_strategy


async def dynamic_round_dialogue(rag, prompt, power_consumption,
                                 c_comfort, strategy, conversation_history,
                                 simulation_env,
                                 timestamp,
                                 mode,
                                 max_rounds_initial=5,
                                 max_rounds_limit=20,
                                 improvement_threshold=0.01,
                                 ):
    """
    """
    current_max_rounds = max_rounds_initial  # 当前允许的轮数上限
    total_rounds_done = 0  # 已完成轮数
    no_improve_count = 0  # 连续未改善计数
    loop_prompt = f"""
                ---Strategy---
                通过你给出的参数调整方案，我们的出了新的系统能耗= {power_consumption}KW/h，
                得到系统的舒适度 = {c_comfort}。这似乎还有改进的空间！请结合历史对话记录和知识库中的内容探索更优秀的策略，
                ---Goal---
                输出格式仍然如下json格式,不要给出多余信息保证输出的完整性：
                {{  “T_chws": <chilled water supply temperature>, 
                            “f_cwp": chilling pump frequency>,
                            “tower_num": <number of cooling towers>.}}

            """
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
            response = await rag.aquery(prompt, param=QueryParam(mode=mode, history_turns=10,
                                                                 conversation_history=conversation_history))
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": response})
            print(response)
            strategy = update_strategy_from_response(response, strategy)
            print("第一次更新的策略：", strategy)
            print("能耗为", power_consumption, "  舒适度为 ", c_comfort)
            combined_val = combine_energy_comfort(power_consumption, c_comfort, alpha=0.5, beta=0.5)
            print(combined_val)
        else:
            round_idx = total_rounds_done + 1
            response = await rag.aquery(loop_prompt, param=QueryParam(mode=mode, history_turns=10,
                                                                      conversation_history=conversation_history))
            conversation_history.append({"role": "user", "content": loop_prompt})
            conversation_history.append({"role": "assistant", "content": response})

            # 保持最多10轮对话（20条记录）
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

            strategy_current = update_strategy_from_response(response, strategy)
            print("优化后的策略:", strategy_current)
            power_consumption, c_comfort = apply_optimized_strategy(simulation_env, timestamp, strategy_current)
            print(f"优化后的能耗:{power_consumption}，优化后的舒适度{c_comfort}")
            combined_val = combine_energy_comfort(power_consumption, c_comfort, alpha=0.5, beta=0.5)
            print(combined_val)

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


async def call_rag_model(rag, conversation_history, chiller_system_info, outdoor_info, strategy):

    prompt = f"""
            您是一个专业的空调冷机系统调优专家，你的任务是结合知识库中的信息和我给出的环境信息对当前空调冷机系统进行优化。
            ---Indoor Env info---
            空调冷机系统的参数如下：Clc是系统冷负荷，T_wet是湿球温度，T_chwr是冷机回水温度，T_chws是冷机供水温度，
            f_cwp是冷却泵频率，Tower_num是冷却塔的个数,f_tower是冷却塔的频率。具体的数值如下所示：
            {json.dumps(chiller_system_info, ensure_ascii=False)}
            ---Outdoor Env info---
            目前的室外环境参数如下：timesamp是当前的时间戳，temperature是室外温度，humidity是室外湿度，precipitation是室外降水量
            wind_speed是室外风速，pressure是室外气压。
            {json.dumps(outdoor_info, ensure_ascii=False)}
            ---Strategy---
            当前的控制参数为：{json.dumps(strategy, ensure_ascii=False)}，在此参数下该空调冷机的系统能耗为{power_consumption}KW/h
            目前的舒适度={c_comfort}(满分1，最差0)
            ---Goal---
            请调整Strategy下的三个控制参数，探索能耗更低且舒适度适中的方案。返回值如下json格式不要给出多余信息。
            {{  “T_chws": <chilled water supply temperature>, 
                “f_cwp": chilling pump frequency>,
                “tower_num": <number of cooling towers>.}}
    """
    return prompt, await rag.aquery(prompt, param=QueryParam(mode='global', history_turns=10,
                                                             conversation_history=conversation_history))


async def main():
    model_path = ""
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
        conversation_history = []
        timestamp = row['timestamp']
        simulation_env.read_environment_parameters(n, FILE_PATH)
        chiller_system_info = {
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
        T_out, Rh_out, precip, wind_speed, pressure = get_weather_data(
            '',
            timestamp)

        outdoor_info = {
            "temperature": T_out,
            "humidity": Rh_out,
            "precipitation": precip,
            "wind_speed": wind_speed,
            "pressure": pressure
        }

        # Initialize RAG instance
        rag = await initialize_rag()

        power_consumption, c_comfort = apply_optimized_strategy(simulation_env, timestamp, strategy)

        prompt = f"""
                    您是一个专业的空调冷机系统调优专家，你的任务是结合知识库中的信息和我给出的环境信息对当前空调冷机系统进行优化。
                    ---Indoor Env info---
                    空调冷机系统的参数如下：Clc是系统冷负荷，T_wet是湿球温度，T_chwr是冷机回水温度，T_chws是冷机供水温度，
                    f_cwp是冷却泵频率，Tower_num是冷却塔的个数,f_tower是冷却塔的频率。具体的数值如下所示：
                    {json.dumps(chiller_system_info, ensure_ascii=False)}
                    ---Outdoor Env info---
                    目前的室外环境参数如下：timesamp是当前的时间戳，temperature是室外温度，humidity是室外湿度，precipitation是室外降水量
                    wind_speed是室外风速，pressure是室外气压。
                    {json.dumps(outdoor_info, ensure_ascii=False)}
                    ---Strategy---
                    
                    T_chws可调控精度为0.3，调控范围是7.0至12.0
                    tower_num可调控精度为1，调控范围是1至4
                    f_cwp可调控精度为0.2，调控范围是32.0至50
                    
                    当前的控制参数为：{json.dumps(strategy, ensure_ascii=False)}，
                
                    在此参数下该空调冷机的系统能耗为{power_consumption}KW/h
            
                    目前的舒适度={c_comfort}(满分1，最差0)
                    ---Goal---
                    请调整Strategy下的三个控制参数，探索能耗更低且舒适度适中的方案。返回值如下json格式不要给出多余信息。
                    {{  “T_chws": <chilled water supply temperature>, 
                        “f_cwp": chilling pump frequency>,
                        “tower_num": <number of cooling towers>.}}
            """

        best_metric, best_info = await dynamic_round_dialogue(rag, prompt, power_consumption, c_comfort, strategy,
                                                              conversation_history,
                                                              simulation_env, timestamp,
                                                              mode='mix',
                                                              max_rounds_initial=10,
                                                              max_rounds_limit=20,
                                                              improvement_threshold=0.01,
                                                              )
        energy = best_info[0]  # 能耗
        comfort = best_info[1]  # 舒适度
        metric = best_metric  # 指标
        # 追加写入数据行
        with open(RESULT_PATH, 'a', newline='') as f:  # 注意用 'a' 追加模式
            writer = csv.writer(f)
            writer.writerow([timestamp, energy, comfort, metric])


def update_knowledge():
    # 设置文件夹路径，存放所有的 txt 文件
    folder_path = ""

    # 顺序（按文件名排序）读取所有 txt 文件内容
    texts = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                texts.append(content)

    rag = asyncio.run(initialize_rag())
    print('初始化完成')
    rag.insert(texts)


if __name__ == "__main__":
    asyncio.run(main())
    # update_knowledge()
