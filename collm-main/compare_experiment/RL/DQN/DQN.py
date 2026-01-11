
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

import ENV_for_rl
import pandas as pd
import numpy as np
env_data = pd.read_csv('/home/lgy/Experiment/Code/Simulation_Model/data_with_weather.csv')
action_all = pd.read_csv('/home/lgy/Experiment/Code/Simulation_Model/action_all_no_decimal.csv')
action_all = np.array(action_all)

env = ENV_for_rl.ChillerEnv(env_data=env_data, action_all=action_all)


model = DQN(MlpPolicy, env, policy_kwargs=dict(net_arch=[256, 128, 64]),
            verbose=1,buffer_size=10000, exploration_fraction=0.3,
            exploration_final_eps=0.2,exploration_initial_eps=1.0,
            batch_size=128, learning_rate=0.0003,gamma=0.98)
print("开始学习")
model.learn(total_timesteps=102640, log_interval=4,progress_bar=True)
print("学习完成")


model.save("dqn_chiller-")
columns = ['Clc', 'T_wet', 'reward', 'timestamp', 'energy_consumption', 'T_chwr', 'T_cwr', 'T_cws', 'Comfort']
data_df = pd.DataFrame(columns=columns)
model = DQN.load("dqn_chiller")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # 提取obs中的Clc和T_wet值，假设obs是一个字典或者可以通过索引访问这些值
    Clc = obs[0]  # 根据实际情况修改
    T_wet = obs[1]  # 根据实际情况修改

    # 提取info中的特定参数
    timestamp = info.get('timestamp')
    # P_chiller = info.get('P_chiller')
    # P_cwp = info.get('P_cwp')
    # P_tower = info.get('P_tower')
    T_chwr = info.get('T_chwr')
    T_cwr = info.get('T_cwr')
    T_cws = info.get('T_cws')
    comfort = info.get('Comfort')
    energy_consumption=info.get('energy_consumption')
    # 将数据添加到DataFrame中
    data_df.loc[len(data_df)] = [Clc, T_wet, reward, timestamp, energy_consumption,T_chwr, T_cwr, T_cws,comfort]
    print(obs, reward, terminated, truncated, info)
    if terminated or truncated:
        # 当环境终止或截断时重置并考虑是否在此处保存数据
        obs, info = env.reset()
        # 可选择在这里调用data_df.to_csv(...)来保存数据
        data_df.to_csv('output_.csv', index=False)
        break
