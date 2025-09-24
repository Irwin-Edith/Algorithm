import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 设置文件夹路径
base_path = 'D:\VS Repository\Algorithm\Data_operated'
subfolders = ['Cardboard', 'Sponge', 'Towel']

# 存储最终数据的列表
data_list = []
labels = []

# 遍历每个子文件夹，读取数据并处理
for label_idx, subfolder in enumerate(subfolders):
    subfolder_path = os.path.join(base_path, subfolder)
    
    # 获取子文件夹中的所有csv文件
    csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(subfolder_path, csv_file)
        
        # 读取CSV文件
        data = pd.read_csv(file_path)
        
        # 删除不需要的列：时间戳和合力列
        sensor_data = data.drop(columns=["Timestamp", "1-2-1x1-X", "1-2-1x1-Y", "1-2-1x1-Z"])

        # 重塑数据为 (样本数, 120, 3)
        sensor_data = sensor_data.values.reshape(-1, 120, 3)

        # 标准化数据（每个方向单独标准化）
        scaler = StandardScaler()
        sensor_data = sensor_data.reshape(-1, 3)  # 重塑为一维数组进行标准化
        sensor_data = scaler.fit_transform(sensor_data)
        sensor_data = sensor_data.reshape(-1, 120, 3)  # 重新重塑为120x3
        
        # 将数据加入到列表中
        data_list.append(sensor_data)
        
        # 为该样本分配标签（根据文件夹名称）
        labels.append(label_idx)  # 0 for Cardboard, 1 for Sponge, 2 for Towel

# 将所有数据合并为一个大的数组
data_array = np.concatenate(data_list, axis=0)  # 形状 (样本数, 120, 3)
labels_array = np.array(labels)  # 形状 (样本数, )

# 将标签数组扩展为每个表面类型的 60 个样本
expanded_labels = np.repeat(labels_array, 120)  # 每个样本有 120 个测点，因此每个标签扩展 120 次

# 检查标签长度是否和数据长度匹配
print(f"expanded_labels 长度: {expanded_labels.shape[0]}")
print(f"数据的长度: {data_array.shape[0]}")

# 将数据合并为一个 DataFrame
final_data = pd.DataFrame(data_array.reshape(len(data_array), -1))  # 转换为二维格式
final_data['label'] = expanded_labels  # 添加标签列

# 查看数据处理结果
final_data.head()

# ...existing code...
final_data.to_csv('final_data.csv', index=False)  # 保存到当前目录
# ...existing code...