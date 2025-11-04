import pandas as pd

# 读取您的两个 CSV 文件
# 请确保这两个 CSV 文件和您的 Python 脚本在同一个文件夹下
df1 = pd.read_csv('/home/yangy50/project/pywasphe/PyWasPhe/whole_feature_final_fibrosis.csv')
df2 = pd.read_csv('/media/yangy50/Elements/KPMP_new/whole_feature_final_fibrosis.csv')

# 获取两个文件各自的列名
df1_columns = set(df1.columns)
df2_columns = set(df2.columns)

# 找出两个文件共有的列名
common_columns = list(df1_columns.intersection(df2_columns))

# 筛选出两个文件中共有的列
df1_filtered = df1[common_columns]
df2_filtered = df2[common_columns]

# 将两个文件合并（增加行）
merged_df = pd.concat([df1_filtered, df2_filtered], ignore_index=True)

# 将合并后的结果保存到一个新的 CSV 文件中
merged_df.to_csv('/media/yangy50/Elements/KPMP_new/whole_feature_final_fibrosis_all_5_batch.csv', index=False)

print("文件合并成功！合并后的文件名为 'merged_features.csv'")