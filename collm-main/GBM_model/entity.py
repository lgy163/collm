from collections import OrderedDict

from numpy import array


class ChillerU:
    def __init__(self, chiller_model, P_ref, C_ref, T_cwr, F_cw, F_chw):

        self.chillerID = 0
        self.powerState = False
        self.chiller_model = chiller_model
        self.P_ref = P_ref
        self.C_ref = C_ref
        self.C_lc = 0.0
        self.T_chwr = 0.0
        self.T_chws = 0.0
        self.F_cw = F_cw
        self.F_chw = F_chw
        self.T_cwr = T_cwr

    def resetT_chwr(self, T_chwr):
        self.T_chwr = T_chwr

    def resetT_cwr(self, T_cwr):
        self.T_cwr = T_cwr

    def resetF_cw(self, F_cw):
        self.F_cw = F_cw

    def controllerState(self, C_lc):
        self.C_lc = C_lc

    def setT_chws(self, T_chws):
        self.T_chws = T_chws

    def getPLR(self):
        return self.C_lc / self.C_ref

    def getT_cwr(self):
        return self.T_cwr

    def calcT_chwr(self):
        self.T_chwr = self.T_chws + self.C_lc / ((4.2 * 1000 * self.F_chw) / 3600)
        return self.T_chwr

    def calcT_cws(self):
        return self.T_cwr + (self.calcP_chiller() + self.C_lc) / (
                (4.2 * self.F_cw * 1000) / 3600
        )

    def getT_chwr(self):
        return self.T_chwr

    def calcP_chiller(self):
        if self.powerState:
            # 拟合的参数是当前的P_chiller/P_ref
            return self.P_ref * self.chiller_model.predict(array([[self.T_chws, self.getT_cwr(), self.getPLR()]]))[0]
        else:
            return 0.0


class PumpU:
    def __init__(self, P_ref, flow_ref, f_ref, P_pump_parameters):
        """
        初始化水泵基本参数
        :param P_ref: 水泵额定功率
        :param flow_ref: 水泵额定水流量
        :param f_ref: 水泵额定频率
        :param P_pump_parameters: 水泵功率计算参数
        """
        self.PumpID = 0  # Assuming PumpID initialization
        self.P_ref = P_ref
        self.Flow_ref = flow_ref
        self.Flow_cl = 0.0
        self.f_ref = f_ref
        self.f_cl = 0.0
        self.P_pump_parameters = P_pump_parameters

    def controller_state(self, f_cl):
        """
        设定水泵当前状态
        (当前水泵工作流量可以经由当前工作频率算出)

        :param f_cl: 水泵当前工作频率
        """
        self.Flow_cl = (f_cl / self.f_ref) * self.Flow_ref
        self.f_cl = f_cl

    def calcF_cw(self):
        """
        计算水泵当前工作流量

        :return: 当前工作流量
        """
        return (self.f_cl / self.f_ref) * self.Flow_ref

    def calcP_pump(self):
        """
        计算水泵能耗

        :return: 水泵能耗
        """
        return self.P_pump_parameters.iloc[0] + self.f_cl * self.P_pump_parameters.iloc[1] + self.f_cl ** 2 * \
            self.P_pump_parameters.iloc[2]


class TowerU:
    def __init__(self, P_ref, Flow_ref, f_ref, approach_model, P_tower_parameters):
        """
        设定冷却塔的基本参数和回归系数
        :param P_ref: 冷却塔额定功率
        :param Flow_ref: 冷却塔额定水流量
        :param f_ref: 冷却塔额定频率
        :param approach_model: Approach回归模型
        :param P_tower_parameters: 冷却塔能耗回归系数
        """
        self.towerID = 0
        self.P_ref = P_ref
        self.Flow_ref = Flow_ref
        self.f_ref = f_ref
        self.f_cl = 0.0  # 冷却塔风机的当前工作频率
        self.step_len = 0.2  # 探索的步长
        self.T_cws = 0.0  # 冷却水供水温度|即冷却塔进水
        self.T_wet = 0.0  # 室外湿球温度（实测）
        self.approach_model = approach_model
        self.P_tower_parameters = P_tower_parameters

    def resetT_cws(self, T_cws):
        """
        重置冷却塔冷却水供水温度

        :param T_cws: 冷却水供水温度
        """
        self.T_cws = T_cws

    def controllerState(self, f_cl, T_wet):
        """
        设定冷却塔当前状态

        :param f_cl: 冷却塔当前的工作频率
        :param T_wet: 冷却塔当前的户外湿球温度
        """
        self.f_cl = f_cl
        self.T_wet = T_wet

    def getFlow_ref(self):
        """
        获取冷却塔额定水流量

        :return: 冷却塔额定水流量
        """
        return self.Flow_ref

    def getT_cws(self):
        """
        获取冷却塔冷却水供水温度
        :return: 冷却塔冷却水供水温度
        """
        return self.T_cws

    def getFRair(self):
        """
        FRair:空气流量比(事实的空气流量比/设计的空气流量比) （风机的频率比值 * 额定量）

        :return: FRair
        """
        return self.f_cl / self.f_ref

    def calcP_tower(self):
        """
        计算冷却塔能耗

        :return: P_tower
        """
        return self.P_tower_parameters.iloc[0] + self.f_cl * self.P_tower_parameters.iloc[1] + self.f_cl ** 2 * \
            self.P_tower_parameters.iloc[2]

    def calcApproach(self, FR_water, T_cwr):
        """
        计算Approach

        :param FR_water: 冷却塔对应的水泵水流量比
        :param T_cwr: 冷却水回水温度
        :return: approach
        """
        FR_air = self.getFRair()
        Tr = self.T_cws - T_cwr
        T_wb = self.T_wet
        approach = self.approach_model.predict(array([[FR_air, FR_water, T_wb, Tr]]))[0]
        return approach

    def calcError(self, FR_water):
        """
        计算Error

        :param FR_water: 冷却塔对应的水泵水流量比
        :return: [min_error, T_cwr]
        """
        T_cwr = 0
        tm = OrderedDict()  # ERRORs,T_cwr
        for i in range(int(50 / self.step_len)):
            T_cwr += self.step_len
            approach = self.calcApproach(FR_water, T_cwr)
            error = abs(T_cwr - (approach + self.T_wet))
            tm[error] = T_cwr

        return tm[min(tm.keys())]
