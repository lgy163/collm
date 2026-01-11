import pandas as pd

# 原 CSV 路径
file_path = r'D:\pyproject\pythonProject\compare_experiment\RL\DQN\output.csv'

# 1. 读取文件
df = pd.read_csv(file_path)

# 2. 计算 COP，并添加为新列
#    如果 Clc 列中可能有 0，请根据需要处理除零情况
df['cop'] = df['Clc']/df['energy_consumption']

# 3. 覆盖写回原文件
df.to_csv(file_path, index=False)

print("已在原文件中添加 ‘cop’ 列，并保存至：", file_path)
