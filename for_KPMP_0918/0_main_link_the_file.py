import os
import pandas as pd
from bin_tool import get_KPMP_id_associate
import shutil

# This file is to organize the data
root_path="/media/yangy50/Elements/KPMP_new/"
origin_data="origin/"
link_file="link_file.csv"
result_path="perpared_data_topology/"

# get the association id
wsi_ls = [f.name for f in os.scandir(root_path+origin_data)]

# read the whole dataframe
link_df = pd.read_csv(root_path+link_file)

# For every kpmp_folder, find the participant id
for wsi_id in wsi_ls:
    matched_row=get_KPMP_id_associate(wsi_id,link_df)
    if len(matched_row)==0:
        continue
    patient_id=matched_row.iloc[0,0]
    # make the folder
    os.makedirs(root_path+result_path+patient_id+"/"+wsi_id,exist_ok=True)

    if os.path.exists(root_path + origin_data + wsi_id + "/Pathomic/final_combined_features_tubules.xlsx"):
        # copy the tubur csv to the folder
        shutil.copy(root_path + origin_data + wsi_id + "/Pathomic/final_combined_features_tubules.xlsx",
                    root_path+result_path + patient_id +"/"+ wsi_id + "/tubules_Features.xlsx")
    else:
        print(wsi_id)
        print("tublues")

    if os.path.exists(root_path + origin_data + wsi_id + "/Pathomic/final_combined_features_non_globally_sclerotic_gloms.xlsx"):
        # copy the tubur csv to the folder
        shutil.copy(root_path + origin_data + wsi_id + "/Pathomic/final_combined_features_non_globally_sclerotic_gloms.xlsx",
                    root_path+result_path + patient_id +"/"+ wsi_id + "/non_globally_sclerotic_glomeruli_Features.xlsx")
    else:
        print(wsi_id)
        print("non_globally_sclerotic_gloms")

    if os.path.exists(root_path + origin_data + wsi_id + "/Pathomic/final_combined_features_muscular_vessels.xlsx"):
        # copy the tubur csv to the folder
        shutil.copy(root_path + origin_data + wsi_id + "/Pathomic/final_combined_features_muscular_vessels.xlsx",
                    root_path+result_path + patient_id +"/"+ wsi_id + "/arteriesarterioles_Features.xlsx")
    else:
        print(wsi_id)
        print("muscular_vessels")
    if os.path.exists(root_path + origin_data + wsi_id + "/Pathomic/final_combined_features_globally_sclerotic_gloms.xlsx"):
        # copy the tubur csv to the folder
        shutil.copy(root_path + origin_data + wsi_id + "/Pathomic/final_combined_features_globally_sclerotic_gloms.xlsx",
                    root_path+result_path + patient_id +"/"+ wsi_id + "/globally_sclerotic_glomeruli_Features.xlsx")
    else:
        print(wsi_id)
        print("globally_sclerotic_gloms")
