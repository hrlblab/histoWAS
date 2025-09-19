# This file is to combine the excel features and change to csv

import pandas as pd
from functools import reduce
import os

root_path="/media/yangy50/Elements/KPMP_new/"
input_path="perpared_data_topology/"
link_file="link_file.csv"
result_path="concate_csv_topology/"

# read all the folder
patient_ls = [f.name for f in os.scandir(root_path+input_path)]

xlsx_file="tubules_Features.xlsx"

for patient in patient_ls:
    wsi_id_ls=[f.name for f in os.scandir(root_path+input_path+patient)]
    # create a new result_csv to append all the wsi to one csv
    result_df=[]
    for wsi_id in wsi_id_ls:
        dfs = pd.read_excel(root_path+input_path+patient+"/"+wsi_id+"/"+xlsx_file, sheet_name=["Features","Bounding Boxes"])
        # dfs_ls=[dfs["Features"], dfs["Bounding Boxes"]]
        features_df = dfs["Features"]
        bboxes_df = dfs["Bounding Boxes"]
        csv_df = pd.merge(features_df, bboxes_df, left_on="compartment_id", right_on="compartment_ids", how="inner")


        csv_df["wsi_id"]=wsi_id
        result_df.append(csv_df.copy())
    result_csv = pd.concat(result_df, axis=0, ignore_index=True)
    result_csv['topology_x'] = (result_csv['x1'] + result_csv['x2']) / 2
    result_csv['topology_y'] = (result_csv['y1'] + result_csv['y2']) / 2
    result_csv.drop(columns=["compartment_ids"], axis=1,inplace=True)
    os.makedirs(root_path+result_path+patient, exist_ok=True)
    # save the result_csv
    result_csv.to_csv(root_path+result_path+patient+"/tubules_Features.csv", index=False)




print(1)