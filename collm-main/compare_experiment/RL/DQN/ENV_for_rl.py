"""
强化学习环境：
1. 状态空间[可观测环境]：冷机冷负荷，湿球温度，冷冻水进水口温度，冷却水回水温度，冷却水出水口温度
{CLc, T_wet, T_chwr, T_cwr, T_cws}
2. 动作空间：冷冻水出水口温度，冷却泵频率，冷却塔台数，冷机开关状态
{T_chws, f_cwp, tower_num, is_chiller_open}
"""
import numpy as np
from gymnasium import Env, spaces
from GroupModel import GroupModel
from compare_experiment.Ablation_Study.Only_LLM.LLM import comfort_score_debug

info_list_env = []


class ChillerEnv(Env):
    def __init__(self, env_data, action_all):
        # 初始化父类
        super(ChillerEnv, self).__init__()
        self.action_all = action_all
        # 定义离散动作空间
        self.action_space = spaces.Discrete(len(action_all))

        # 定义观测空间
        self.observation_space = spaces.Box(
            low=np.array([env_data['CLc'].min(), env_data['T_wet'].min()], dtype=np.float32),
            high=np.array([env_data['CLc'].max(), env_data['T_wet'].max()], dtype=np.float32),
            dtype=np.float32
        )

        # 初始化冷机模型
        self.chiller = GroupModel(ref_parameter={
            'P_ref': 534.8,  # 冷机额定功率
            'COP_ref': 5.65,  # 冷机额定工况COP
            'C_ref': 534.8 * 5.65,  # 冷机额定制冷量
            'P_ch_pump_ref': 90,  # 冷却泵额定功率
            'Flow_cp_ref': 800,  # 冷却泵额定水流量
            'f_cp_ref': 50,  # 冷却泵额定频率
            'P_tower_ref': 11,  # 单台冷却塔额定功率
            'Flow_tower_ref': 1400,  # 冷却塔额定水流量
            'f_tower_ref': 50  # 冷却塔额定频率
        }, model_path='/home/lgy/Experiment/Code/Simulation_Model/save_model')

        self.env_data = env_data  # 环境数据
        self.current_time_index = 0  # 当前时间索引
        self.reward_val = 0  # 奖励值
        self.CLc = 0  # 当前冷负荷
        self.T_wet = 0  # 当前湿球温度
        self.action_index = 0  # 当前动作索引
        self.action_continuous_list = []  # 动作连续值列表

    def step(self, action):
        """
        根据action计算环境数据
        :param action: 离散动作索引
        :return: next_state, reward, done, truncated, info
        """
        action_continuous = self.discretize_action(action)

        self.action_continuous_list.append(action_continuous)
        if self.env_data is None:
            raise ValueError("未设置环境数据.")
        else:
            # 获取当前时刻的环境数据
            f_chwp_list = self.env_data[['f_chwp1', 'f_chwp2', 'f_chwp3', 'f_chwp4', 'f_chwp5', 'f_chwp6']].iloc[
                self.current_time_index]
            # 计算冷水机组各个设备功率以及冷却水回水温度和冷却水出水口温度
            P_chiller, T_chwr, T_cwr, T_cws, P_cwp, P_tower, F_cw = self.chiller.get_result(
                CLc=self.CLc,
                T_chws=action_continuous[0],
                T_wet=self.T_wet,
                f_cwp=action_continuous[1],
                tower_num=action_continuous[2],
                f_chwp_list=f_chwp_list,
                is_chiller_open=True if self.CLc > 500 else False
            )
            T_out, Rh_out, precip, wind_speed, pressure = self.env_data['温度'],self.env_data['湿度'],self.env_data['降水量'],self.env_data['风速'],self.env_data['气压']
            T_out = T_out.iloc[self.current_time_index]
            Rh_out =  Rh_out.iloc[self.current_time_index]
            precip = precip.iloc[self.current_time_index]
            wind_speed = wind_speed.iloc[self.current_time_index]
            pressure = pressure.iloc[self.current_time_index]
            # 计算奖励
            c_comfort = comfort_score_debug(T_chwr, T_out, Rh_out, precip, wind_speed, pressure)


            reward = self.reward(self.CLc, P_chiller, P_cwp, P_tower,c_comfort, self.T_wet, F_cw)
            self.reward_val = reward

            info = {
                'timestamp': self.env_data['timestamp'].iloc[self.current_time_index],
                'CLc': self.CLc,
                'T_chws_action': action_continuous[0],
                'f_cwp_action': action_continuous[1],
                'tower_num_action': action_continuous[2],
                'P_chiller': P_chiller,
                'P_cwp': P_cwp,
                'P_tower': P_tower,
                'T_chwr': T_chwr,
                'T_cwr': T_cwr,
                'T_cws': T_cws,
                'Comfort': c_comfort,
                'COP_s': reward,
                'energy_consumption': P_chiller+P_cwp+P_tower,
            }
            info_list_env.append(info)

            # 状态指针后移
            self.current_time_index += 1
            # 动作指针后移
            self.action_index += 1

            # 判断是否结束
            done = self.current_time_index >= len(self.env_data) - 1
            truncated = False  # Gymnasium 新增的 "truncated" 标志，表示是否由于时间限制而终止

            # 获取下一时刻的环境数据
            if not done:
                self.CLc = self.env_data['CLc'].iloc[self.current_time_index]
                self.T_wet = self.env_data['T_wet'].iloc[self.current_time_index]

            next_state = np.array([self.CLc, self.T_wet], dtype=np.float32)
            return next_state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """
        Gymnasium 的 reset 方法需要返回初始状态和额外信息
        """
        super().reset(seed=seed)
        self.current_time_index = 0
        self.CLc = self.env_data['CLc'].iloc[self.current_time_index]
        self.T_wet = self.env_data['T_wet'].iloc[self.current_time_index]
        initial_state = np.array([self.CLc, self.T_wet], dtype=np.float32)
        info = {}  # 可选返回的初始化信息

        # 重置动作列表索引
        self.action_index = 0
        return initial_state, info

    def reward(self, CLc, P_chiller, P_cwp, P_tower,c_comfort, T_wet, flow_tower):
        """
        计算奖励值
        """
        # COPs = CLc / (P_chiller + P_cwp + P_tower) if (P_chiller + P_cwp + P_tower) > 0 else 0


        comfort = c_comfort
        energy = P_chiller + P_cwp + P_tower
        energy_score = (730.0 - energy) / 480.0  # 480 = 730 - 250
        energy_score = max(0.0, min(1.0, energy_score))
        # 舒适度直接使用，并确保在 [0,1] 范围内
        comfort_score = max(0.0, min(1.0, comfort))
        # 综合评价: 数值越大表示综合效果越好
        combined_val = 0.3 * energy_score + 0.7 * comfort_score

        # changes = 0
        # # 设定每十二条数据是否变动一次冷却塔台数，变动超过两次则给与惩罚
        # if len(self.action_continuous_list) > 12:
        #     tower_num_list = [action[2] for action in self.action_continuous_list[-12:]]
        #     if len(set(tower_num_list)) > 4:
        #         changes = -1
        #     else:
        #         changes = 0
        # if len(self.action_continuous_list) > 2:
        #     # 计算动作连续性惩罚，即上一个动作和当前动作的差值是否超过阈值
        #     last_action = self.action_continuous_list[self.action_index - 1]
        #     current_action = self.action_continuous_list[self.action_index]
        #     if last_action[2] - current_action[2] > 1:
        #         # 增加冷却塔台数的惩罚
        #         reward -= 0.5
        # reward = COPs + comfort_score * 0.05 + changes
        return combined_val

    def render(self):
        pass

    def close(self):
        pass

    def discretize_action(self, action_index):
        action_values = self.action_all[action_index]
        return action_values

    def inverse_discretize(self, continuous_action):
        discrete_index = np.argmin(np.linalg.norm(self.action_all - continuous_action, axis=1))
        return discrete_index

