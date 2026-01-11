from typing import List

from container import ChillerC, PumpC, TowerC
from entity import ChillerU, PumpU, TowerU


class Model:
    def __init__(self, model: dict, parameter_tower_pump, parameter):
        '''
        初始化模型
        :param model: 模型字典
        :param parameter: 参数字典
        '''
        # 加载模型
        print(model.keys())
        chiller_model = model['P_chiller']
        approach_model = model['Approach']
        print(parameter_tower_pump)
        pump_model = parameter_tower_pump.loc['P_pump']
        tower_model = parameter_tower_pump.loc['P_tower']

        # 冷机额定参数
        Chill_P_ref = parameter['P_ref']
        Chill_C_ref = parameter['C_ref']
        Pump_P_ref_cool = parameter['P_ch_pump']
        Pump_F_chwRef_cool = parameter['Flow_cp_ref']
        Pump_f_ref = parameter['f_cp_ref']
        Tower_P_ref = parameter['P_tower_ref']
        Tower_F_ref = parameter['Flow_tower_ref']
        Tower_f_ref = parameter['f_tower_ref']

        F_chw = 38  # 冷冻水流量 这里根据冷机的铭牌参数设置（冷机的额定冷冻水流量）
        F_cw = 30  # 冷却水流量 这里根据冷机的铭牌参数设置（冷机的额定冷却水流量）
        T_cwr = 30  # 冷却水回水温度

        # 初始化设备，注意其中的T_chwr、f_pump、f_tower、T_wet参数为初始化设定值，后续会根据输入数据组进行更新迭代
        # 目前只开了一个冷水机组，包括冷机，冷却水泵，冷却塔
        chiller_1 = ChillerU(chiller_model, Chill_P_ref, Chill_C_ref, T_cwr, F_cw, F_chw)
        chiller_2 = ChillerU(chiller_model, Chill_P_ref, Chill_C_ref, T_cwr, F_cw, F_chw)
        cooling = PumpU(Pump_P_ref_cool, Pump_F_chwRef_cool, Pump_f_ref, pump_model)
        tower = TowerU(Tower_P_ref, Tower_F_ref, Tower_f_ref, approach_model, tower_model)

        # 初始化容器
        pumpC = PumpC()
        towerC = TowerC()
        # 添加设备到容器
        pumpC.add(cooling)
        towerC.add(tower)
        # 设置容器设备状态
        pumpC.setState(30)
        towerC.setState(30, 25)
        # 装载冷机组设备
        self.chillerC = ChillerC(pumpC, towerC)
        self.chillerC.add(chiller_1)
        self.chillerC.add(chiller_2)
        # 设置冷机初始冷负荷和电源状态
        self.chillerC.setState([0,0], [False,False])

    # 获取每步仿真的结果
    def get_result(self, ratio: List[float], CLs, T_chws: List[float], f_pump, f_tower, T_wet):
        '''
        输入数据组：[ratio], CLs, f_pump, f_tower, T_wet
            ratio: 为各个冷机设备分配的冷负荷比例 [1]
            CLs: 系统总冷负荷
            T_chws: 冷冻水出水温度 (考虑到各个冷机的设置不同，所以使用列表)
            f_pump: 冷却水泵频率
            f_tower: 冷却塔风扇频率
            T_wet: 湿球温度
        输出数据组：P_chiller, P_pump, P_tower, T_chwr, T_cwr, T_cws
            P_chiller: 冷机功率
            P_pump: 冷却水泵功率
            P_tower: 冷却塔功率
            T_chwr: 冷冻水回水温度
            T_cwr: 冷却水回水温度
            T_cws: 冷却水出水温度
        '''
        return self.chillerC.data_step(ratio, CLs, T_chws, f_pump, f_tower, T_wet)


