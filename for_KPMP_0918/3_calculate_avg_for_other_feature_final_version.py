import pandas as pd
import os

# This file is to caculate the avg of the feature

result_path = "/media/yangy50/Elements/KPMP_new/"
wsi_data_path = "/media/yangy50/Elements/KPMP_new/concate_csv_topology/"
whole_feature_path = "/media/yangy50/Elements/KPMP_new/Discribe_score.csv"
result_file_name="whole_feature_final_fibrosis.csv"

# --- 1. 读取 whole_feature csv 并按 Biopsy ID 聚合 ---
# read the whole_feature csv
whole_feature = pd.read_csv(whole_feature_path, usecols=['Biopsy ID: ', 'Interstitial Fibrosis (%)'])


whole_feature['Interstitial Fibrosis (%)'] = pd.to_numeric(
    whole_feature['Interstitial Fibrosis (%)'],
    errors='coerce'
)

# 【修改点 1】: 按 'Biopsy ID: ' 分组，并计算所有数值列（这里是'ACR (mg/g)'）的平均值
# numeric_only=True 确保只对数字列求平均，避免非数字列出错
whole_feature_avg = whole_feature.groupby('Biopsy ID: ').mean(numeric_only=True).reset_index()

# --- 2. 读取 topology_feature 并按 Biopsy ID 聚合 ---
topology_feature = pd.read_csv("/media/yangy50/Elements/KPMP_new/topology_feature_final.csv")

# 【修改点 2】: 按 'Biopsy ID: ' 分组，并计算所有数值特征列的平均值
topology_feature_avg = topology_feature.groupby('Biopsy ID: ').mean(numeric_only=True).reset_index()

# --- 3. 循环计算 WSI 特征的平均值 (这部分逻辑保留) ---
# This is the result ls to save the final features
result_ls = []

# 注意：这里我们遍历 "whole_feature_avg" 来获取唯一的 Biopsy ID 列表
for index, row in whole_feature_avg.iterrows():
    biopsy_id = row['Biopsy ID: ']

    # 检查 WSI 特征文件是否存在
    tubules_feature_file = os.path.join(wsi_data_path, biopsy_id, "tubules_Features.csv")

    if not os.path.exists(tubules_feature_file):
        # 如果文件不存在，可以选择跳过，或者添加一个带NaN的空行
        # 这里我们选择跳过，以匹配原始逻辑
        print(f"Skipping {biopsy_id}: File not found at {tubules_feature_file}")
        continue

    # read the biopsy file
    wsi_data_df = pd.read_csv(tubules_feature_file)

    # This is for batch 2-5
    # (假设你已经注释了正确的版本，我们使用 batch 1 的版本)
    #column_means = wsi_data_df.drop(columns=["compartment_id","wsi_id","x1","y1","x2","y2","topology_x","topology_y","In Medulla"]).mean(skipna=True)
    cols_to_drop_batch1 = ["compartment_id","wsi_id","x1","y1","x2","y2","topology_x","topology_y","In Medulla"]

    # This is for batch 1
    # 确保列存在才删除
    # cols_to_drop_batch1 = ["compartment_ids", "wsi_id", "x1", "y1", "x2", "y2", "topology_x", "topology_y"]
    # 筛选出wsi_data_df中实际存在的列进行drop
    existing_cols_to_drop = [col for col in cols_to_drop_batch1 if col in wsi_data_df.columns]
    column_means = wsi_data_df.drop(columns=existing_cols_to_drop).mean(skipna=True)

    column_means["Biopsy ID: "] = biopsy_id
    # save the series
    result_ls.append(column_means.to_frame().T)

# 检查 result_ls 是否为空，如果为空则无法继续
if not result_ls:
    print("Error: No valid WSI feature files were found or processed. Exiting.")
else:
    # change the result_ls to a dataframe
    wsi_result_df = pd.concat(result_ls,
                              axis=0,  # 纵向合并
                              ignore_index=True)

    # --- 4. 合并所有已经聚合的 DataFrame ---

    # 1. wsi_result_df (WSI平均特征) combine with whole_feature_avg (ACR平均特征):
    # 使用 'inner' merge，只保留三者共有的 Biopsy ID
    result_csv = pd.merge(whole_feature_avg, wsi_result_df, on='Biopsy ID: ', how='inner')

    # 2. combine with topology_feature_avg (拓扑平均特征)
    result_csv = pd.merge(result_csv, topology_feature_avg, on='Biopsy ID: ', how='inner')

    # --- 5. 保存最终结果 ---
    output_file = os.path.join(result_path, result_file_name)
    result_csv.to_csv(output_file, index=False)

    print(f"Successfully merged {len(result_csv)} unique Biopsy IDs.")
    print(f"Final file saved to: {output_file}")

print(1)  # 原始代码中的结束标志