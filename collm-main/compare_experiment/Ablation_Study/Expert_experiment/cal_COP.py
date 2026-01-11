import pandas as pd

# 定义文件路径
file_path = "Expert_Experiment_september_result.csv"
CLc_file_path = r"D:\pyproject\pythonProject\Simulation_Model\data.csv"  # 注意使用原始字符串(r)
output_path = "september_with_cop.csv"  # 输出文件路径

# 读取数据
try:
    # 读取主文件
    df_energy = pd.read_csv(file_path, parse_dates=['timestamp'])

    # 读取CLc文件
    df_clc = pd.read_csv(CLc_file_path, parse_dates=['timestamp'])

except FileNotFoundError as e:
    print(f"文件未找到: {e}")
    exit()
except KeyError as e:
    print(f"列名错误: 请确认CSV文件中存在'timestamp'列")
    exit()

# 合并数据（按时间戳对齐）
merged_df = pd.merge(
    df_energy[['timestamp', 'energy_consumption']],
    df_clc[['timestamp', 'CLc']],  # 假设CLc值所在列名为"CLc"
    on='timestamp',
    how='inner'  # 只保留两者都有的时间戳
)

# 计算COp
try:
    merged_df['cop'] = merged_df['CLc'] / merged_df['energy_consumption']
except KeyError as e:
    print(f"列名错误: 请确认CSV文件中存在'CLc'和/或'energy_consumption'列")
    exit()
except ZeroDivisionError:
    print("警告：发现energy_consumption为0的值，已替换为NaN")
    merged_df['cop'] = merged_df['CLc'] / merged_df['energy_consumption'].replace(0, pd.NA)

# 保存结果
merged_df.to_csv(output_path, index=False)
print(f"结果已保存至 {output_path}")