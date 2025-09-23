import pandas as pd
import os

# 定义输入和输出根目录
input_root = r'd:\VS Repository\Algorithm\Data9_17'
output_root = r'd:\VS Repository\Algorithm\Data_operated'

# 需要处理的子文件夹
folders = ['Cardboard', 'Sponge', 'Towel']

for folder in folders:
    input_dir = os.path.join(input_root, folder)
    output_dir = os.path.join(output_root, folder)
    os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            df = pd.read_csv(input_path)
            df_filtered = df.iloc[40:160]  # 保留42~162行
            df_filtered.to_csv(output_path, index=False)