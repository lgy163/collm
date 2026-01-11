import pandas as pd
from model import Model
import os
from joblib import dump, load

# 参数映射
parameter_mapping = {
    'P_ref': 40.8,  # 冷机额定功率
    'COP': 5.03,  # 冷机额定工况COP
    'C_ref': 40.8 * 5.03,  # 冷机额定制冷量
    'P_ch_pump': 2.2,  # 冷却泵额定功率
    'Flow_cp_ref': 45,  # 冷却泵额定水流量
    'f_cp_ref': 50,  # 冷却泵额定频率
    'P_tower_ref': 4,  # 冷却塔额定功率
    'Flow_tower_ref': 82,  # 冷却塔额定水流量
    'f_tower_ref': 50  # 冷却塔额定频率
}

# 模型路径
model_path = '/home/lgy/Experiment/Code/GBM_model/save_model'
model = {}
parameter_tower_pump = None

# 读取路径下的模型
for file in os.listdir(model_path):
    if file.endswith('.plk'):
        # 处理模型文件，去掉文件后缀
        model_name = file.split('.')[0]
        model_file_path = os.path.join(model_path, file)
        print(f"正在加载模型: {model_file_path}")
        model[model_name] = load(model_file_path)

        # 检查加载的模型类型
        print(f"加载的 {model_name} 模型类型: {type(model[model_name])}")

    elif file.endswith('.csv'):
        # 处理参数文件
        parameter_file_path = os.path.join(model_path, file)
        print(f"正在加载参数文件: {parameter_file_path}")
        parameter_tower_pump = pd.read_csv(parameter_file_path, index_col=0)

# 检查是否成功加载了参数
if parameter_tower_pump is None:
    raise Exception('参数读取失败！')

# 检查是否成功加载了 P_chiller 模型
if 'P_chiller' not in model.keys():
    raise Exception('P_chiller 模型读取失败！')

# 初始化 Model 类
c_model = Model(model=model, parameter_tower_pump=parameter_tower_pump, parameter=parameter_mapping)

# 仿真输入参数
ratio = [0.8,0.2]  # 冷机负荷分配比例（假设只有一个冷机，负荷比例为100%）
CLs = 800.0  # 系统总冷负荷 (kW)
T_chws = [7.0,6.0]  # 冷冻水出水温度 (℃)，假设只有一个冷机
f_pump = 60.0  # 冷却水泵的频率 (Hz)
f_tower = 60.0  # 冷却塔风机频率 (Hz)
T_wet = 33.0  # 室外湿球温度 (℃)

# 获取仿真结果
result = c_model.get_result(ratio, CLs, T_chws, f_pump, f_tower, T_wet)

# 输出结果
if result is not None:
    for i, res in enumerate(result):
        print(f"冷机 {i + 1} 仿真结果:")
        print(f"冷机功率: {res[0]} kW")
        print(f"冷却泵功率: {res[1]} kW")
        print(f"冷却塔功率: {res[2]} kW")
        print(f"冷冻水回水温度: {res[3]} °C")
        print(f"冷却水回水温度: {res[4]} °C")
        print(f"冷却水出水温度: {res[5]} °C")
else:
    print("仿真未成功，未能获取结果。")


