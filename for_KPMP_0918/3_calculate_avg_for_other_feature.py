import pandas as pd
import os

# This file is to caculate the avg of the feature

result_path="/media/yangy50/Elements/KPMP_new/"
wsi_data_path="/media/yangy50/Elements/KPMP_new/concate_csv_topology/"
whole_feature_path="/media/yangy50/Elements/KPMP_new/Discribe_score.csv"

# read the whole_feature csv
whole_feature = pd.read_csv(whole_feature_path, usecols=['Biopsy ID: ', 'Interstitial Fibrosis (%)'])

# read the topology feature
topology_feature=pd.read_csv("/media/yangy50/Elements/KPMP_new/topology_feature.csv")


# This is the result ls to save the final features
result_ls=[]

for index, row in whole_feature.iterrows():
    biopsy_id = row['Biopsy ID: ']
    # if the csv file exists
    if not os.path.exists(wsi_data_path+biopsy_id+"/tubules_Features.csv"):
        continue
    # read the biopsy file
    wsi_data_df=pd.read_csv(wsi_data_path+biopsy_id+"/tubules_Features.csv")
    column_means = wsi_data_df.drop(columns=["compartment_id","wsi_id","x1","y1","x2","y2","topology_x","topology_y","In Medulla"]).mean(skipna=True)
    column_means["Biopsy ID: "]=biopsy_id
    # save the series
    result_ls.append(column_means.to_frame().T)

# change the result_ls to a dataframe
wsi_result_df = pd.concat(result_ls,
               axis=0,  # 纵向合并
               ignore_index=True)

# wsi_result_df combine with whole_feature:
result_csv = pd.merge(whole_feature, wsi_result_df, on='Biopsy ID: ', how='inner')
result_csv=pd.merge(result_csv,topology_feature,on='Biopsy ID: ', how='inner')
# save the result_csv

result_csv.to_csv(result_path+"final_features.csv", index=False)

print(1)
