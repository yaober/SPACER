import os
import pandas as pd
import numpy as np

base_dir = 'validation_tumor/tumor_antigen_developmental_stage/tedd_finish'

def process_organ_folder(folder_path):
    organ_name = os.path.basename(folder_path)
    print(f"Processing: {organ_name}")

    count_path = os.path.join(folder_path, 'counts.csv')
    if not os.path.exists(count_path):
        count_path += '.gz'
        if not os.path.exists(count_path):
            print(f"Skipped {organ_name}: No count file found")
            return

    # 读取表达矩阵并转置
    count = pd.read_csv(count_path)
    count = count.T

    meta_path = os.path.join(folder_path, 'meta.csv')
    if not os.path.exists(meta_path):
        print(f"Skipped {organ_name}: No meta file")
        return
    meta = pd.read_csv(meta_path, index_col=0)

    # 合并 meta 信息
    count = count.merge(meta, left_index=True, right_index=True, how='left')

    # 分出 fetal 和 adult 数据
    fetal = count[count['Timepoint'].str.startswith('GW')]
    adult = count[~count['Timepoint'].str.startswith('GW')]

    # 选择数值列（表达矩阵列）
    expr_cols = fetal.select_dtypes(include=[np.number]).columns

    # 计算平均表达
    fetal_mean = fetal[expr_cols].mean(axis=0)
    adult_mean = adult[expr_cols].mean(axis=0)

    # 保存结果
    fetal_mean.to_csv(os.path.join(folder_path, 'new_fetal_mean_expr.csv'), header=True)
    adult_mean.to_csv(os.path.join(folder_path, 'new_adult_mean_expr.csv'), header=True)

for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
    if os.path.isdir(subpath):
        process_organ_folder(subpath)
