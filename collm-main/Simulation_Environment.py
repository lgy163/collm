# agents/SimulationEnvironment.py

import os
import pandas as pd
from joblib import load


class SimulationEnvironment:
    def __init__(self, ref_parameter, model_path, initial_parameters=None):
        """
        初始化仿真环境
        :param ref_parameter: 字典，包含参考参数
        :param model_path: 模型文件路径
        :param initial_parameters: 字典，包含初始参数，可选
        """
        self.model_path = model_path
        # 铭牌参数
        self.P_ref = ref_parameter['P_ref']
        self.C_ref = ref_parameter['C_ref']
        self.P_ch_pump_ref = ref_parameter['P_ch_pump_ref']
        self.Flow_cp_ref = ref_parameter['Flow_cp_ref']
        self.f_cp_ref = ref_parameter['f_cp_ref']
        self.P_tower_ref = ref_parameter['P_tower_ref']
        self.Flow_tower_ref = ref_parameter['Flow_tower_ref']
        self.f_tower_ref = ref_parameter['f_tower_ref']

        # 环境数据
        self.CLc = None  # 冷机冷负荷
        self.T_wet = None  # 湿球温度
        self.T_chwr = None  # 冷冻水回水温度
        self.f_chwp_list = None

        # 控制数据
        self.T_chws = None  # 冷冻水供水温度
        self.f_cwp = None  # 冷却泵频率
        self.tower_num = None  # 冷却塔台数

        self.f_tower = 50  # 冷却塔频率50, 定频

        # 输出数据
        self.P_chiller = None  # 冷机功率
        self.T_cwr = self.T_wet  # 系统循环启动变量
        self.T_cws = 30  # 系统循环启动变量
        self.P_c_pump = None  # 冷却泵功率
        self.P_tower = None  # 冷却塔功率

        # 冷却水流量
        self.F_cw = 0

        # 探索步长
        self.step_len = 0.2

        # 计算模型
        self.chiller_model = None
        self.T_cwr_model = None
        self.T_cws_model = None
        self.F_carrier_model = None
        self.P_c_pump_model = pd.DataFrame(columns=['a0', 'a1', 'a2', 'a3'])  # 模型系数
        self.load_model()

        # 初始化参数（如果提供）
        if initial_parameters:
            self.set_simulation_parameters(initial_parameters)

    def load_model(self):
        model = {}
        # 读取路径下的模型
        for file in os.listdir(self.model_path):
            if file.endswith('.plk'):
                # 去掉文件后缀
                model_name = file.split('.')[0]
                model[model_name] = load(os.path.join(self.model_path, file))
            if file.endswith('.csv'):
                # 去掉文件后缀
                model_name = file.split('.')[0]
                if model_name == 'P_c_pump':
                    self.P_c_pump_model = pd.read_csv(os.path.join(self.model_path, file), index_col=0)

        if model:
            self.chiller_model = model['chiller']
            self.T_cwr_model = model['T_cwr']
            self.T_cws_model = model['T_cws']
            self.F_carrier_model = model['F_carrier']
        else:
            raise ValueError("No model found in the path.")

    def get_P_chiller(self, T_chws, T_cwr, PLR):
        """
        计算冷机功率
        :param T_chws: 冷冻水供水温度
        :param T_cwr: 冷却水回水温度
        :param PLR: 冷机负荷比
        :return: 冷机功率
        """
        # 示例系数
        b1, b2, b3, b4, b5, b6, d1, d2, d3, d4, d5, d6, g1, g2, g3 = [
            982.6275439260432,
            -81.25273008384187,
            -0.0022962965984289862,
            9.10204977866597,
            -0.490731175122823,
            0.6002741205658229,
            -148.14943296261464,
            22.709543798533446,
            -0.88275538208516,
            4.773390577943974,
            -0.043995829211765265,
            -0.387907189112693,
            -0.00017265546619449107,
            0.00014547510039078363,
            -0.00026020758200679897
        ]

        # 用函数替代模型
        def DOE(data, b1, b2, b3, b4, b5, b6, d1, d2, d3, d4, d5, d6, g1, g2, g3):
            P_ref = self.P_ref
            T_chws, T_cwr, PLR = data
            chillerCapFTemp = b1 + b2 * T_chws + b3 * T_chws ** 2 + b4 * T_cwr + b5 * T_cwr ** 2 + b6 * T_cwr * T_chws
            chillerEIREFTemp = d1 + d2 * T_chws + d3 * T_chws ** 2 + d4 * T_cwr + d5 * T_cwr ** 2 + d6 * T_cwr * T_chws
            chillerRIRFPLR = g1 + g2 * PLR + g3 * PLR ** 2
            return P_ref * chillerCapFTemp * chillerEIREFTemp * chillerRIRFPLR

        if PLR < 0.2:
            return 0
        else:
            P_chiller = DOE([T_chws, T_cwr, PLR], b1, b2, b3, b4, b5, b6, d1, d2, d3, d4, d5, d6, g1, g2, g3)
        return P_chiller

    def get_P_c_pump(self, f_cwp):
        a0, a1, a2, a3 = self.P_c_pump_model.loc['LR']
        if f_cwp < 32:
            return 0
        else:
            return a0 + a1 * f_cwp + a2 * f_cwp ** 2 + a3 * f_cwp ** 3

    def get_P_tower(self, tower_num):
        return self.P_tower_ref * int(tower_num)

    def get_T_cws(self, CLc, T_chws, PLR, T_wet):
        # 包装成 df 格式
        data = pd.DataFrame(columns=['CLc', 'T_chws', 'PLR', 'T_wet'])
        data.loc[0] = [CLc, T_chws, PLR, T_wet]
        return self.T_cws_model.predict(data)[0]

    def get_T_cwr(self, T_cws, P_tower, T_wet, F_cw):
        data = pd.DataFrame(columns=['T_cws', 'P_tower', 'T_wet', 'F_cw'])
        data.loc[0] = [T_cws, P_tower, T_wet, F_cw]
        # 将预测值类型转换为 float
        return self.T_cwr_model.predict(data)[0]

    def get_F_carrier(self, CLc, f_chwp1, f_chwp2, f_chwp3, f_chwp4, f_chwp5, f_chwp6):
        Chp_num = 0
        for i in [f_chwp1, f_chwp2, f_chwp3, f_chwp4, f_chwp5, f_chwp6]:
            if i > 0:
                Chp_num += 1
        data = pd.DataFrame(
            columns=['CLc', 'f_chwp1', 'f_chwp2', 'f_chwp3', 'f_chwp4', 'f_chwp5', 'f_chwp6', 'Chp_num'])
        data.loc[0] = [CLc, f_chwp1, f_chwp2, f_chwp3, f_chwp4, f_chwp5, f_chwp6, Chp_num]
        return self.F_carrier_model.predict(data)[0]

    def get_T_chwr(self, T_chws, CLc, F_carrier):
        return T_chws + CLc / ((4.192 * F_carrier) / 3.6)

    def get_P(self):
        step = 0
        delta = 2
        self.F_cw = (self.f_cwp / self.f_cp_ref) * self.Flow_cp_ref

        # 计算开利机的冷冻水流量
        F_carrier = self.get_F_carrier(self.CLc, *self.f_chwp_list)
        # 计算冷却水回水温度
        self.T_chwr = self.get_T_chwr(self.T_chws, self.CLc, F_carrier)


        self.P_tower = self.get_P_tower(self.tower_num)

        if self.tower_num == 0:
            # 若冷却塔没开，则进水温度等于回水温度
            self.T_cws = self.get_T_cws(self.CLc, self.T_chws, (self.CLc / self.C_ref), self.T_wet)
            self.T_cwr = self.T_cws
            self.P_chiller = self.get_P_chiller(self.T_chws, self.T_cwr, (self.CLc / self.C_ref))
        else:
            while delta ** 2 > 0.1 and step < 1000:
                step += 1
                # 计算冷却水回水温度
                T_cwr_calc = self.get_T_cwr(self.T_cws, self.P_tower, self.T_wet, self.F_cw / self.tower_num)
                # 计算冷机功率
                self.P_chiller = self.get_P_chiller(self.T_chws, T_cwr_calc, (self.CLc / self.C_ref))
                # 计算冷却水供水温度
                T_cws_calc = self.get_T_cws(self.CLc, self.T_chws, (self.CLc / self.C_ref), self.T_wet)
                delta = T_cws_calc - self.T_cws
                self.T_cws = T_cws_calc
                self.T_cwr = T_cwr_calc

        self.P_c_pump = self.get_P_c_pump(self.f_cwp)
        return self.P_chiller, self.T_cwr, self.T_cws, self.P_c_pump, self.P_tower, self.F_cw

    def set_control_parameters(self,T_chws, f_cwp,tower_num):
        self.T_chws = T_chws
        self.f_cwp = f_cwp
        self.tower_num = tower_num

    def read_environment_parameters(self, n, csv_filepath):
        """
        Reads the nth row from the given CSV file using pandas and assigns the parameters to instance variables.

        Parameters:
        - n (int): Zero-based index of the row to read.
        - csv_filepath (str): Path to the CSV file.
        """
        try:
            df = pd.read_csv(csv_filepath)
            # Check if n is within the bounds
            if n < 0 or n >= len(df):
                raise IndexError(f"The CSV file does not contain row number {n}.")

            row = df.iloc[n]

            # Assigning scalar parameters
            self.CLc = float(row['CLc'])
            self.T_wet = float(row['T_wet'])
            self.T_chwr = float(row['T_chwr'])
            self.P_chiller = float(row['P_chiller'])
            self.P_tower = float(row['P_tower'])
            self.P_c_pump = float(row['P_cwp'])

            # Assigning list of pump frequencies
            self.f_chwp_list = [
                float(row['f_chwp1']),
                float(row['f_chwp2']),
                float(row['f_chwp3']),
                float(row['f_chwp4']),
                float(row['f_chwp5']),
                float(row['f_chwp6'])
            ]

            # Assigning control data
            self.T_chws = float(row['T_chws'])
            self.f_cwp = float(row['f_cwp'])
            self.tower_num = int(row['Tower_num'])

            print('成功读取第',n,'行')

        except FileNotFoundError:
            print(f"Error: The file '{csv_filepath}' was not found.")
        except KeyError as e:
            print(f"Error: Missing column in CSV file - {e}")
        except ValueError as e:
            print(f"Error: Invalid data type encountered - {e}")
        except IndexError as e:
            print(e)


    def read_environment_parameters_for_fine_tuning_LLM(self, n, csv_filepath):
        """
        Reads the nth row from the given CSV file using pandas and assigns the parameters to instance variables.

        Parameters:
        - n (int): Zero-based index of the row to read.
        - csv_filepath (str): Path to the CSV file.
        """
        try:
            df = pd.read_csv(csv_filepath)
            # Check if n is within the bounds
            if n < 0 or n >= len(df):
                raise IndexError(f"The CSV file does not contain row number {n}.")

            row = df.iloc[n]

            # Assigning scalar parameters
            self.CLc = float(row['CLc'])
            self.T_wet = float(row['T_wet'])
            self.P_chiller = float(row['P_chiller'])
            self.P_tower = float(row['P_tower'])
            self.P_c_pump = float(row['P_cwp'])

            # Assigning list of pump frequencies
            self.f_chwp_list = [
                float(row['f_chwp1']),
                float(row['f_chwp2']),
                float(row['f_chwp3']),
                float(row['f_chwp4']),
                float(row['f_chwp5']),
                float(row['f_chwp6'])
            ]

            # Assigning control data
            self.T_chws = float(row['T_chws'])
            self.f_cwp = float(row['f_cwp'])
            self.tower_num = int(row['tower_num'])

            print(f"Environment parameters successfully read from row {n}.")

        except FileNotFoundError:
            print(f"Error: The file '{csv_filepath}' was not found.")
        except KeyError as e:
            print(f"Error: Missing column in CSV file - {e}")
        except ValueError as e:
            print(f"Error: Invalid data type encountered - {e}")
        except IndexError as e:
            print(e)

    def get_result(self, CLc, T_chws, T_wet, f_cwp, tower_num, f_chwp_list, is_chiller_open):
        """
        :param CLc: 冷机系统冷负荷，这里看作冷机当前冷负荷
        :param T_wet: 湿球温度
        :param T_chws: 冷冻水供水温度
        :param f_cwp: 冷却泵频率
        :param tower_num: 冷却塔台数
        :param f_chwp_list: 冷冻泵频率列表
        :param is_chiller_open: 冷机是否开启
        :return:
            P_chiller: 冷机功率
            T_cwr: 冷却水回水温度
            T_cws: 冷却水出水温度
            P_c_pump: 冷却泵功率
            P_tower: 冷却塔功率
            F_cw: 冷却塔功率
        """
        self.CLc = CLc
        self.T_chws = T_chws
        self.T_wet = T_wet
        self.f_cwp = f_cwp
        self.tower_num = tower_num
        self.f_chwp_list = f_chwp_list
        if is_chiller_open:
            return self.get_P()
        else:
            return 0, 0, 0, 0, 0, 0


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
    # 环境初始参数

    model_path = "../Simulation_Model/save_model"  # 模型文件路径
    simulation = SimulationEnvironment(ref_parameter,model_path)
    simulation.read_environment_parameters()
