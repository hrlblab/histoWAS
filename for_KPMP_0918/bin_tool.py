



def get_KPMP_id_associate(kpmp_wsi_id,link_df):
    matched_rows = link_df[link_df["Slide_name"].str.contains(kpmp_wsi_id, na=False, case=False)]
    return matched_rows