from typing import List, Union

from entity import ChillerU, PumpU, TowerU


class PumpC(PumpU):
    def __init__(self):
        """
        冷却泵容器构造方法，预先初始化冷却泵设备对象列表
        """
        self.f_cl = 0.0  # 冷却泵工作频率
        self.count = 0  # 冷却泵设备对象数量
        self.pumpList = []  # 冷却泵设备对象列表

    def add(self, pumpU: PumpU):
        """
        向冷却泵容器中添加一个冷却泵设备对象

        :param pumpU: 冷却泵设备对象
        """
        pumpU.PumpID = self.count
        self.pumpList.append(pumpU)
        self.count += 1

    def setState(self, f_cl):
        """
        设定各冷却泵状态

        :param f_cl: 冷却泵的工作频率
        """
        self.f_cl = f_cl
        for pumpU in self.pumpList:
            pumpU.controller_state(self.f_cl)

    def getInstance(self) -> PumpU:
        """
        获取冷却泵设备对象的实例
        多个冷却泵是一体的，获取单个的冷却泵数据即为整体的冷却泵数据

        :return: 冷却泵设备对象
        """
        pump = self.pumpList[0] if self.pumpList else None
        if pump:
            return pump
        else:
            print("容器为空")
            return None

    def get_count(self):
        return self.count

    def calcF_cw(self, chiller_num):
        """
        算的应该是单个冷机分配到的流量
        总流量/冷机数量

        :param chiller_num: 冷机数量
        :return: 单个冷机分配到的流量
        """
        return self.calcSumF_cw() / chiller_num

    def calcSumF_cw(self):
        """
        计算冷却泵总水流量

        :return: 冷却泵总水流量
        """
        sum_F_cw = 0.0
        for pumpU in self.pumpList:
            sum_F_cw += pumpU.calc_F_cw()
        return sum_F_cw

    def calcP_pump(self):
        """
        计算所有冷却泵功率总和

        :return: 冷却泵功率总和
        """
        sum_p_pump = 0.0
        for pumpU in self.pumpList:
            sum_p_pump += pumpU.calc_p_pump()
        return sum_p_pump


class TowerC(TowerU):
    def __init__(self):
        self.f_cl = 0.0  # 所有冷却塔工作频率
        self.T_wet = 0.0  # 所有冷却塔对应的户外湿球温度
        self.towerUList = []  # 冷却塔设备对象列表

    count = 0  # 冷却塔设备对象数量

    def getInstance(self):
        """
        多个冷却塔是一体的，获取单个的冷却塔数据即为整体的冷却塔数据
        :return: TowerU tower 单个冷却塔设备对象
        """
        if self.towerUList:
            return self.towerUList[0]
        else:
            print("容器为空")
            return None

    def add(self, towerU):
        """
        向冷却塔容器中添加一个冷却塔设备对象
        :param towerU: 待添加的冷却塔设备对象
        """
        towerU.towerID = self.count
        self.towerUList.append(towerU)
        self.count += 1

    def setState(self, f_cl, T_wet):
        """
        设定冷却塔的设备状态
        :param f_cl: 冷却塔工作频率
        :param T_wet: 冷却塔的室外湿球温度
        """
        self.f_cl = f_cl
        self.T_wet = T_wet
        for towerU in self.towerUList:
            towerU.controllerState(self.f_cl, self.T_wet)

    def getCount(self):
        """
        获取冷却塔容器中的冷却塔设备对象数量
        :return: int count 冷却塔设备对象数量
        """
        return self.count

    def calcP_tower(self):
        """
        计算所有冷却塔功率总和
        :return: double sumP_tower 冷却塔功率总和
        """
        sumP_tower = 0.0
        for towerU in self.towerUList:
            sumP_tower += towerU.calcP_tower()
        return sumP_tower


class ChillerC(ChillerU):
    def __init__(self, pumpC: PumpC, towerC: TowerC):
        self.chillerlist: List[ChillerU] = []  # List to store chiller objects
        # 各个冷机的冷负荷列表
        self.ClcList: Union[None, List[float]] = None
        # 各个冷机的开关机状态列表
        self.powerList: Union[None, List[bool]] = None

        # 冷却泵容器对象
        self.pumpC = pumpC
        # 冷却塔容器对象
        self.towerC = towerC

        self.T_chwr = 12  # Initialize T_chwr
        self.T_cwr = 30  # Initialize T_cwr
        self.T_cws: List[float] = []  # List to store T_cws for each chiller

        # 冷机数量
        self.count: int = 0  # Initialize chiller count

    def add(self, chillerU: ChillerU):
        '''
        添加冷机设备对象
        :param chillerU: 冷机设备对象
        '''
        if chillerU is None:
            print("添加的设备对象为空！")
            return
        chillerU.chillerID = self.count
        self.chillerlist.append(chillerU)
        self.count += 1

    def getCount(self):
        return self.count

    def setState(self, clcList: List[float], powerList: List[bool]):
        '''
        设定各个冷机的状态，顺序添加，与添加冷机设备对象的顺序一致
        :param clcList: 各个冷机的冷负荷列表
        :param powerList:  各个冷机的开关机状态列表
        :return:
        '''
        self.ClcList = clcList
        self.powerList = powerList
        for i in range(len(self.chillerlist)):
            chillerU = self.chillerlist[i]
            chillerU.powerState = self.powerList[i]
            chillerU.controllerState(self.ClcList[i])

    def data_step(self, ratio, CLs, T_chws, f_pump, f_tower, T_wet):
        if len(ratio) != self.count:
            raise ValueError("负荷比例列表长度与冷机数量不匹配")
        if sum(ratio) != 1:
            raise ValueError("负荷比例总和不为1")

        clcList = [CLs * r for r in ratio]
        powerList = [r != 0 for r in ratio]
        self.setState(clcList, powerList)
        self.pumpC.setState(f_pump)
        self.towerC.setState(f_tower, T_wet)
        for i in range(len(self.chillerlist)):
            self.chillerlist[i].setT_chws(T_chws[i])
        return self.calcResult()

    def calcResult(self) -> Union[None, List[List[float]]]:
        '''
        计算仿真结果
        :return:
        '''
        if self.count == 0:
            print("系统未添加冷机")
            return None
        else:
            re = []
            for i in range(len(self.chillerlist)):
                chillerU = self.chillerlist[i]
                # 对每个冷水机组进行仿真并获取仿真结果
                re.append(self.run(chillerU))
            return re

    def run(self, chiller: ChillerU):
        if chiller.powerState:
            converge = []
            step = 0
            Tchwr_chiller = 0.0
            while True:
                Tchwr_chiller = chiller.calcT_chwr()
                chiller.resetT_chwr(Tchwr_chiller)
                self.towerC.getInstance().resetT_cws(chiller.calcT_cws())
                chiller.resetF_cw(self.pumpC.getInstance().calcF_cw())
                T_cwr = self.towerC.getInstance().calcError(
                    self.pumpC.getInstance().calcF_cw() / self.towerC.getInstance().getFlow_ref())
                chiller.resetT_cwr(T_cwr)
                converge.append(T_cwr)
                if (len(converge) >= 2 and abs(converge[step] - converge[step - 1]) <= 0.1) or step >= 50:
                    if step >= 50:
                        print("超出50步")
                    break
                step += 1
            P_pump = self.pumpC.getInstance().calcP_pump()
            P_tower = self.towerC.getInstance().calcP_tower()

            # P_chiller, P_pump, P_tower, T_chwr, T_cwr, T_cws

            return [chiller.calcP_chiller(), P_pump, P_tower, Tchwr_chiller, chiller.getT_cwr(),
                    self.towerC.getInstance().getT_cws()]

        else:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
