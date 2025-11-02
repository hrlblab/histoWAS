import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, skew, kurtosis
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# =============================================================================
# 1. RPY2 和 spatstat 设置
# =============================================================================
try:
    base = importr('base')
    stats = importr('stats')
    spatstat_geom = importr('spatstat.geom')
    spatstat_explore = importr('spatstat.explore')
    print("rpy2 和 spatstat 包加载成功。")
except Exception as e:
    print(f"错误: R 或 spatstat 包未能成功加载: {e}")
    print("请确保您已在R中安装 'spatstat' (install.packages('spatstat'))")
    exit()

# 忽略Pandas的FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# 2. 从 2_calculate_topology_features...py 复制的常量
# =============================================================================

# 设定密度图的像素大小
DENSITY_MAP_PIXEL_SIZE = 5.0


# =============================================================================
# 3. 复制的核心特征函数
# (来自 2_calculate_topology_features...py 和 test_robustness...py)
# =============================================================================

# --- K/L-function (来自 test_robustness...py, 已为模拟优化) ---
def calculate_k_features_r_backend(area_df: pd.DataFrame,
                                   combined_df: pd.DataFrame,
                                   process_list: list,
                                   distance_list: list,
                                   if_L: bool = True) -> pd.DataFrame:
    """
    [R后端版] K-function 和 L-function.
    """
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    results_list = []

    for wsi_id, group_df in combined_df.groupby('wsi_id'):
        wsi_result = {'wsi_id': wsi_id}
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            try:
                if len(group_df) < 2:
                    for d in distance_list:
                        wsi_result[f"{object_type_1}_{object_type_2}_distance_{d}_L_function"] = np.nan
                    results_list.append(wsi_result)
                    continue

                # *** 模拟关键: 使用 area_df 中的精确窗口 ***
                area_info = area_df.loc[area_df['wsi_id'] == wsi_id]
                win_dims = area_info['window_dims'].iloc[0]

                r.assign("x_min_r", win_dims[0])
                r.assign("x_max_r", win_dims[1])
                r.assign("y_min_r", win_dims[2])
                r.assign("y_max_r", win_dims[3])
                r.assign("distance_list", distance_list)

                points1_df = group_df[group_df['object_type'] == object_type_1]
                r.assign("points1_x_r", points1_df['topology_x'].values)
                r.assign("points1_y_r", points1_df['topology_y'].values)

                # 在 r=... 中使用更密集的采样以进行平滑插值
                r_script = f"""
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)
                k_result <- Kest(ppp1, correction="iso", r=seq(0, max(distance_list), length.out=101))
                k_result
                """
                k_result_r = r(r_script)

                r_distances = k_result_r['r']
                r_k_values = k_result_r['iso']

                k_values_at_distances = np.interp(distance_list, r_distances, r_k_values, left=np.nan, right=np.nan)

                if if_L:
                    k_for_l_calc = np.maximum(0, k_values_at_distances)
                    l_values_at_distances = np.sqrt(k_for_l_calc / np.pi) - np.array(distance_list)

                    for i, d in enumerate(distance_list):
                        l_feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_L_function"
                        wsi_result[l_feature_name] = l_values_at_distances[i]

                results_list.append(wsi_result)

            except Exception as e:
                print(f"处理 wsi_id '{wsi_id}' 时发生未知错误: {e}")
                for d in distance_list:
                    if if_L:
                        wsi_result[f"{object_type_1}_{object_type_2}_distance_{d}_L_function"] = np.nan
                results_list.append(wsi_result)

    return pd.DataFrame(results_list) if results_list else pd.DataFrame()


# --- g-function (来自 test_robustness...py, 已为模拟优化) ---
def calculate_g_features_r_backend(area_df: pd.DataFrame,
                                   combined_df: pd.DataFrame,
                                   process_list: list,
                                   distance_list: list) -> pd.DataFrame:
    """
    [R后端版] g-function (Pair Correlation Function).
    """
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    results_list = []

    for wsi_id, group_df in combined_df.groupby('wsi_id'):
        wsi_result = {'wsi_id': wsi_id}
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            try:
                if len(group_df) < 2:
                    for d in distance_list:
                        wsi_result[f"{object_type_1}_{object_type_2}_distance_{d}_g_function"] = np.nan
                    results_list.append(wsi_result)
                    continue

                # *** 模拟关键: 使用 area_df 中的精确窗口 ***
                area_info = area_df.loc[area_df['wsi_id'] == wsi_id]
                win_dims = area_info['window_dims'].iloc[0]

                r.assign("x_min_r", win_dims[0])
                r.assign("x_max_r", win_dims[1])
                r.assign("y_min_r", win_dims[2])
                r.assign("y_max_r", win_dims[3])
                r.assign("distance_list", distance_list)

                points1_df = group_df[group_df['object_type'] == object_type_1]
                r.assign("points1_x_r", points1_df['topology_x'].values)
                r.assign("points1_y_r", points1_df['topology_y'].values)

                # 在 r=... 中使用更密集的采样以进行平滑插值
                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)
                pcf_result <- pcf(ppp1, correction="iso", r=seq(0, max(distance_list), length.out=101))
                pcf_result
                """
                g_result_r = r(r_script)

                r_distances = g_result_r['r']
                r_g_values = g_result_r['iso']

                g_values_at_distances = np.interp(distance_list, r_distances, r_g_values, left=np.nan, right=np.nan)

                for i, d in enumerate(distance_list):
                    g_feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_g_function"
                    wsi_result[g_feature_name] = g_values_at_distances[i]
                results_list.append(wsi_result)

            except Exception as e:
                print(f"处理 wsi_id '{wsi_id}' 时发生错误: {e}")
                for d in distance_list:
                    wsi_result[f"{object_type_1}_{object_type_2}_distance_{d}_g_function"] = np.nan
                results_list.append(wsi_result)

    return pd.DataFrame(results_list) if results_list else pd.DataFrame()


# --- G-function (来自 2_calculate_topology_features...py) ---
def calculate_gest_features_r_backend(area_df: pd.DataFrame,
                                      combined_df: pd.DataFrame,
                                      process_list: list,
                                      distance_list: list) -> pd.DataFrame:
    """
    [R后端版] G-function (最近邻距离分布函数)。
    """
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    is_univariate = (object_type_1 == object_type_2)
    results_list = []

    for wsi_id, group_df in combined_df.groupby('wsi_id'):
        wsi_result = {'wsi_id': wsi_id}
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            try:
                # *** 模拟关键: 使用 area_df 中的精确窗口 ***
                area_info = area_df.loc[area_df['wsi_id'] == wsi_id]
                win_dims = area_info['window_dims'].iloc[0]

                if len(group_df) < 2:
                    for d in distance_list:
                        wsi_result[f"{object_type_1}_{object_type_2}_distance_{d}_G_function"] = np.nan
                    results_list.append(wsi_result)
                    continue

                r.assign("x_min_r", win_dims[0])
                r.assign("x_max_r", win_dims[1])
                r.assign("y_min_r", win_dims[2])
                r.assign("y_max_r", win_dims[3])
                r.assign("distance_list", distance_list)

                points1_df = group_df[group_df['object_type'] == object_type_1]
                n1 = len(points1_df)

                if n1 < 2:  # Gii需要至少2个点
                    for d in distance_list:
                        wsi_result[f"{object_type_1}_{object_type_2}_distance_{d}_G_function"] = np.nan
                    results_list.append(wsi_result)
                    continue

                r.assign("points1_x_r", points1_df['topology_x'].values)
                r.assign("points1_y_r", points1_df['topology_y'].values)

                if is_univariate:
                    r_script = """
                    win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                    ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)
                    Gest(ppp1, correction="km", r=seq(0, max(distance_list), length.out=101))
                    """
                    gest_result_r = r(r_script)
                else:
                    # (为简洁起见，此模拟仅测试单变量，但保留了框架)
                    print("G-function 仅为单变量测试")
                    continue

                r_distances = gest_result_r['r']
                r_gest_values = gest_result_r['km']

                gest_values_at_distances = np.interp(distance_list, r_distances, r_gest_values, left=np.nan,
                                                     right=np.nan)

                for i, d in enumerate(distance_list):
                    g_feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_G_function"
                    wsi_result[g_feature_name] = gest_values_at_distances[i]
                results_list.append(wsi_result)

            except Exception as e:
                print(f"处理 wsi_id '{wsi_id}' (Gest) 时发生错误: {e}")
                for d in distance_list:
                    wsi_result[f"{object_type_1}_{object_type_2}_distance_{d}_G_function"] = np.nan
                results_list.append(wsi_result)

    return pd.DataFrame(results_list) if results_list else pd.DataFrame()


# --- F-function (来自 2_calculate_topology_features...py) ---
def calculate_fest_features_r_backend(area_df: pd.DataFrame,
                                      combined_df: pd.DataFrame,
                                      process_list: list,
                                      distance_list: list) -> pd.DataFrame:
    """
    [R后端版] F-function (空空间函数).
    """
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    results_list = []

    for wsi_id, group_df in combined_df.groupby('wsi_id'):
        wsi_result = {'wsi_id': wsi_id}
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            try:
                area_info = area_df.loc[area_df['wsi_id'] == wsi_id]
                win_dims = area_info['window_dims'].iloc[0]

                if len(group_df) < 2:
                    for d in distance_list:
                        wsi_result[f"{object_type_1}_{object_type_2}_distance_{d}_F_function"] = np.nan
                    results_list.append(wsi_result)
                    continue

                r.assign("x_min_r", win_dims[0])
                r.assign("x_max_r", win_dims[1])
                r.assign("y_min_r", win_dims[2])
                r.assign("y_max_r", win_dims[3])
                r.assign("distance_list", distance_list)

                points_df = group_df[group_df['object_type'] == object_type_2]  # F-func 关注 'to' (j)
                n_points = len(points_df)

                if n_points < 1:
                    for d in distance_list:
                        wsi_result[f"{object_type_1}_{object_type_2}_distance_{d}_F_function"] = np.nan
                    results_list.append(wsi_result)
                    continue

                r.assign("points_x_r", points_df['topology_x'].values)
                r.assign("points_y_r", points_df['topology_y'].values)

                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp_obj <- ppp(x=points_x_r, y=points_y_r, window=win)
                Fest(ppp_obj, correction="km", r=seq(0, max(distance_list), length.out=101))
                """
                fest_result_r = r(r_script)

                r_distances = fest_result_r['r']
                r_fest_values = fest_result_r['km']

                fest_values_at_distances = np.interp(distance_list, r_distances, r_fest_values, left=np.nan,
                                                     right=np.nan)

                for i, d in enumerate(distance_list):
                    f_feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_F_function"
                    wsi_result[f_feature_name] = fest_values_at_distances[i]
                results_list.append(wsi_result)

            except Exception as e:
                print(f"处理 wsi_id '{wsi_id}' (Fest) 时发生错误: {e}")
                for d in distance_list:
                    wsi_result[f"{object_type_1}_{object_type_2}_distance_{d}_F_function"] = np.nan
                results_list.append(wsi_result)

    return pd.DataFrame(results_list) if results_list else pd.DataFrame()


# --- J-function (来自 2_calculate_topology_features...py) ---
def calculate_jest_features_r_backend(area_df: pd.DataFrame,
                                      combined_df: pd.DataFrame,
                                      process_list: list,
                                      distance_list: list) -> pd.DataFrame:
    """
    [R后端版] J-function.
    """
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    is_univariate = (object_type_1 == object_type_2)
    results_list = []

    for wsi_id, group_df in combined_df.groupby('wsi_id'):
        wsi_result = {'wsi_id': wsi_id}
        feature_prefix = f"{object_type_1}_{object_type_2}"

        default_features = {}
        for d in distance_list:
            default_features[f"{feature_prefix}_distance_{d}_J_function"] = np.nan
        default_features[f"{feature_prefix}_distance_avg_J_function"] = np.nan  # 也会测试这个

        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            try:
                area_info = area_df.loc[area_df['wsi_id'] == wsi_id]
                win_dims = area_info['window_dims'].iloc[0]

                if len(group_df) < 2:
                    wsi_result.update(default_features)
                    results_list.append(wsi_result)
                    continue

                r.assign("x_min_r", win_dims[0])
                r.assign("x_max_r", win_dims[1])
                r.assign("y_min_r", win_dims[2])
                r.assign("y_max_r", win_dims[3])
                r.assign("distance_list", distance_list)

                points1_df = group_df[group_df['object_type'] == object_type_1]
                n1 = len(points1_df)

                if is_univariate:
                    if n1 < 2:
                        wsi_result.update(default_features)
                        results_list.append(wsi_result)
                        continue

                    r.assign("points1_x_r", points1_df['topology_x'].values)
                    r.assign("points1_y_r", points1_df['topology_y'].values)

                    r_script = """
                    win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                    ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)
                    Jest(ppp1, correction="km", r=seq(0, max(distance_list), length.out=101))
                    """
                    jest_result_r = r(r_script)
                else:
                    # (为简洁起见，此模拟仅测试单变量)
                    continue

                r_distances = jest_result_r['r']
                r_jest_values = jest_result_r['km']

                jest_values_at_distances = np.interp(distance_list, r_distances, r_jest_values, left=np.nan,
                                                     right=np.nan)

                for i, d in enumerate(distance_list):
                    j_feature_name = f"{feature_prefix}_distance_{d}_J_function"
                    wsi_result[j_feature_name] = jest_values_at_distances[i]

                # 额外测试单值特征
                j_mean_feature_name = f"{feature_prefix}_distance_avg_J_function"
                wsi_result[j_mean_feature_name] = np.nanmean(jest_values_at_distances)

                results_list.append(wsi_result)

            except Exception as e:
                print(f"处理 wsi_id '{wsi_id}' (Jest) 时发生错误: {e}")
                wsi_result.update(default_features)
                results_list.append(wsi_result)

    return pd.DataFrame(results_list) if results_list else pd.DataFrame()


# --- Kinhom / Linhom (来自 2_calculate_topology_features...py) ---
def calculate_Kinhom_features_r_backend(area_df: pd.DataFrame,
                                        combined_df: pd.DataFrame,
                                        process_list: list,
                                        distance_list: list,
                                        if_L: bool = True) -> pd.DataFrame:
    """
    [R后端版] 非齐次 K-function (Kinhom) 和 L-function (Linhom)。
    """
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    is_univariate = (object_type_1 == object_type_2)

    if not is_univariate:
        return pd.DataFrame()  # 仅测试单变量

    results_list = []

    for wsi_id, group_df in combined_df.groupby('wsi_id'):
        feature_prefix = f"{object_type_1}_{object_type_2}"
        wsi_result = {'wsi_id': wsi_id}

        default_features = {}
        for d in distance_list:
            if if_L:
                default_features[f"{feature_prefix}_distance_{d}_Linhom_function"] = np.nan

        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            try:
                area_info = area_df.loc[area_df['wsi_id'] == wsi_id]
                win_dims = area_info['window_dims'].iloc[0]

                points_df = group_df[group_df['object_type'] == object_type_1]
                n_points = len(points_df)

                if n_points < 10:
                    wsi_result.update(default_features)
                    results_list.append(wsi_result)
                    continue

                r.assign("points_x_r", points_df['topology_x'].values)
                r.assign("points_y_r", points_df['topology_y'].values)
                r.assign("x_min_r", win_dims[0])
                r.assign("x_max_r", win_dims[1])
                r.assign("y_min_r", win_dims[2])
                r.assign("y_max_r", win_dims[3])
                r.assign("distance_list", distance_list)

                # R 脚本：在CSR测试中，我们提供 *已知* 的恒定强度 lambda
                # 这
                lambda_const = n_points / ((win_dims[1] - win_dims[0]) * (win_dims[3] - win_dims[2]))
                r.assign("lambda_const_r", lambda_const)

                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp_obj <- ppp(x=points_x_r, y=points_y_r, window=win)

                # 步骤 1: 不估计lambda，而是使用已知的常数lambda
                # (在CSR上测试Kinhom的正确方法)
                lambda_known <- as.im(lambda_const_r, W=win)

                # 步骤 2: 调用 Kinhom
                K_result <- Kinhom(ppp_obj, lambda_known, correction="iso", r=seq(0, max(distance_list), length.out=101))
                K_result
                """
                k_result_r = r(r_script)

                r_distances = k_result_r['r']
                r_k_values = k_result_r['iso']
                r_k_theo = k_result_r['theo']  # 理论值 (pi*r^2)

                k_values_at_distances = np.interp(distance_list, r_distances, r_k_values, left=np.nan, right=np.nan)
                k_theo_at_distances = np.interp(distance_list, r_distances, r_k_theo, left=np.nan, right=np.nan)

                if if_L:
                    k_for_l_calc = np.maximum(0, k_values_at_distances)
                    l_values_at_distances = np.sqrt(k_for_l_calc / np.pi)
                    l_theo_at_distances = np.sqrt(k_theo_at_distances / np.pi)
                    l_centered_values = l_values_at_distances - l_theo_at_distances

                    for i, d in enumerate(distance_list):
                        wsi_result[f"{feature_prefix}_distance_{d}_Linhom_function"] = l_centered_values[i]

                results_list.append(wsi_result)

            except Exception as e:
                print(f"处理 wsi_id '{wsi_id}' (Kinhom) 时发生错误: {e}")
                wsi_result.update(default_features)
                results_list.append(wsi_result)

    return pd.DataFrame(results_list) if results_list else pd.DataFrame()


# --- g_inhom (pcfinhom) (来自 2_calculate_topology_features...py) ---
def calculate_ginhom_features_r_backend(area_df: pd.DataFrame,
                                        combined_df: pd.DataFrame,
                                        process_list: list,
                                        distance_list: list) -> pd.DataFrame:
    """
    [R后端版] 计算非齐次 g-function (pcfinhom)。
    """
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    is_univariate = (object_type_1 == object_type_2)

    if not is_univariate:
        return pd.DataFrame()  # 仅测试单变量

    results_list = []

    for wsi_id, group_df in combined_df.groupby('wsi_id'):
        feature_prefix = f"{object_type_1}_{object_type_2}"
        wsi_result = {'wsi_id': wsi_id}

        default_features = {}
        for d in distance_list:
            default_features[f"{feature_prefix}_distance_{d}_ginhom_function"] = np.nan

        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            try:
                area_info = area_df.loc[area_df['wsi_id'] == wsi_id]
                win_dims = area_info['window_dims'].iloc[0]

                points_df = group_df[group_df['object_type'] == object_type_1]
                n_points = len(points_df)

                if n_points < 10:
                    wsi_result.update(default_features)
                    results_list.append(wsi_result)
                    continue

                r.assign("points_x_r", points_df['topology_x'].values)
                r.assign("points_y_r", points_df['topology_y'].values)
                r.assign("x_min_r", win_dims[0])
                r.assign("x_max_r", win_dims[1])
                r.assign("y_min_r", win_dims[2])
                r.assign("y_max_r", win_dims[3])
                r.assign("distance_list", distance_list)

                lambda_const = n_points / ((win_dims[1] - win_dims[0]) * (win_dims[3] - win_dims[2]))
                r.assign("lambda_const_r", lambda_const)

                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp_obj <- ppp(x=points_x_r, y=points_y_r, window=win)

                lambda_known <- as.im(lambda_const_r, W=win)

                g_result <- pcfinhom(ppp_obj, lambda_known, correction="iso", r=seq(0, max(distance_list), length.out=101))
                g_result
                """
                g_result_r = r(r_script)

                r_distances = g_result_r['r']
                r_g_values = g_result_r['iso']

                g_values_at_distances = np.interp(distance_list, r_distances, r_g_values, left=np.nan, right=np.nan)

                for i, d in enumerate(distance_list):
                    wsi_result[f"{feature_prefix}_distance_{d}_ginhom_function"] = g_values_at_distances[i]

                results_list.append(wsi_result)

            except Exception as e:
                print(f"处理 wsi_id '{wsi_id}' (pcfinhom) 时发生错误: {e}")
                wsi_result.update(default_features)
                results_list.append(wsi_result)

    return pd.DataFrame(results_list) if results_list else pd.DataFrame()


# --- Density Stats (来自 2_calculate_topology_features...py) ---
def calculate_density_stats_r_backend(combined_df: pd.DataFrame,
                                      process_list: list) -> pd.DataFrame:
    """
    [R后端版 - 修正] 计算核密度图的统计特征。
    """
    results_list = []

    # 在模拟中，所有点都在一个WSI中，所以我们只处理那个
    for wsi_id, wsi_df in combined_df.groupby('wsi_id'):
        wsi_result = {'wsi_id': wsi_id}

        # *** 模拟关键: 边界框就是我们的窗口 ***
        area_info = area_df.loc[area_df['wsi_id'] == wsi_id]
        win_dims = area_info['window_dims'].iloc[0]
        x_min, x_max = win_dims[0], win_dims[1]
        y_min, y_max = win_dims[2], win_dims[3]

        for object_type in process_list:
            feature_prefix = f"{object_type}_density"
            default_features = {
                f"{feature_prefix}_std_dev": np.nan,
                f"{feature_prefix}_skewness": np.nan,
                f"{feature_prefix}_kurtosis": np.nan
            }

            points_df = wsi_df[wsi_df['object_type'] == object_type]
            n_points = len(points_df)

            if n_points < 10:
                wsi_result.update(default_features)
                continue

            with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
                try:
                    r.assign("points_x_r", points_df['topology_x'].values)
                    r.assign("points_y_r", points_df['topology_y'].values)
                    r.assign("x_min_r", x_min)
                    r.assign("x_max_r", x_max)
                    r.assign("y_min_r", y_min)
                    r.assign("y_max_r", y_max)
                    r.assign("pixel_size_r", DENSITY_MAP_PIXEL_SIZE)

                    r_script = """
                                        win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                                        ppp_obj <- ppp(x=points_x_r, y=points_y_r, window=win)

                                        # --- 修改开始 ---
                                        # 定义一个更宽的搜索范围, 例如从0.1到窗口的最短边
                                        # (在你的模拟中, 宽度和高度相同, 都是 3000)
                                        max_search_range <- min(x_max_r - x_min_r, y_max_r - y_min_r)
                                        search_range <- c(0.1, max_search_range)
                                        # --- 修改结束 ---

                                        sigma <- tryCatch({
                                          # --- 修改：添加 srange 参数 ---
                                          bw.ppl(ppp_obj, srange=search_range)
                                        }, error = function(e) {
                                          (x_max_r - x_min_r) / 8 
                                        })

                                        D <- density(ppp_obj, sigma=sigma, eps=c(pixel_size_r, pixel_size_r))
                                        v <- as.vector(D$v)
                                        v <- v[is.finite(v)]
                                        v
                                        """
                    density_values_r = r(r_script)
                    density_values_np = np.array(density_values_r)

                    if len(density_values_np) < 4:
                        wsi_result.update(default_features)
                        continue

                    wsi_result[f"{feature_prefix}_std_dev"] = np.std(density_values_np, ddof=1)
                    wsi_result[f"{feature_prefix}_skewness"] = skew(density_values_np, bias=False)
                    wsi_result[f"{feature_prefix}_kurtosis"] = kurtosis(density_values_np, bias=False, fisher=True)

                except Exception as e:
                    print(f"处理 WSI {wsi_id} 的 '{object_type}' (Density Stats) 时发生R错误: {e}")
                    wsi_result.update(default_features)

        results_list.append(wsi_result)

    return pd.DataFrame(results_list)


# --- Global Density (来自 2_calculate_topology_features...py) ---
def calculate_global_density(combined_df: pd.DataFrame,
                             area_df: pd.DataFrame,
                             process_list: list) -> pd.DataFrame:
    """
    计算指定对象的全局平均密度。
    """
    results_list = []
    for index, row in area_df.iterrows():
        wsi_id = row['wsi_id']
        tissue_area = row['tissue_area']  # *** 模拟关键: 使用已知的面积 ***
        wsi_result = {'wsi_id': wsi_id}
        wsi_annotations = combined_df[combined_df['wsi_id'] == wsi_id]

        for object_name in process_list:
            num_objects = (wsi_annotations['object_type'] == object_name).sum()
            feature_name = f"{object_name}_global_density"
            if tissue_area > 0:
                density = num_objects / tissue_area
            else:
                density = 0.0
            wsi_result[feature_name] = density
        results_list.append(wsi_result)

    return pd.DataFrame(results_list)


# --- ANN (来自 2_calculate_topology_features...py) ---
def _calculate_ann(points):
    """辅助函数：计算平均最近邻距离 (ANN)"""
    if points.shape[0] < 2:
        return np.nan
    distance_matrix = pdist(points, 'euclidean')
    square_distance_matrix = squareform(distance_matrix)
    np.fill_diagonal(square_distance_matrix, np.inf)
    nearest_neighbor_distances = np.min(square_distance_matrix, axis=1)
    ann_value = np.mean(nearest_neighbor_distances)
    return ann_value


# --- ANN Index (来自 2_calculate_topology_features...py) ---
def calculate_ann_index_features(combined_df: pd.DataFrame,
                                 area_df: pd.DataFrame,
                                 process_list: list) -> pd.DataFrame:
    """
    计算平均最近邻指数 (ANN Index / Clark-Evans Index)。
    """
    results_list = []
    grouped = combined_df.groupby('wsi_id')

    for wsi_id, group_df in grouped:
        wsi_result = {'wsi_id': wsi_id}

        # *** 模拟关键: 使用已知的面积 ***
        try:
            tissue_area = area_df.loc[area_df['wsi_id'] == wsi_id, 'tissue_area'].iloc[0]
        except IndexError:
            tissue_area = 0

        for obj_type in process_list:
            feature_name = f"ANN_Index_{obj_type}"
            points_df = group_df[group_df['object_type'] == obj_type]
            points = points_df[['topology_x', 'topology_y']].values
            n_points = len(points)

            if n_points < 2 or tissue_area <= 0:
                wsi_result[feature_name] = np.nan
                continue

            observed_ann = _calculate_ann(points)
            lambda_density = n_points / tissue_area
            expected_ann = 1.0 / (2.0 * np.sqrt(lambda_density))

            if expected_ann > 0:
                ann_index = observed_ann / expected_ann
            else:
                ann_index = np.nan
            wsi_result[feature_name] = ann_index
        results_list.append(wsi_result)

    return pd.DataFrame(results_list)


# =============================================================================
# 4. 模拟功能函数 (来自 test_robustness_more_features.py)
# =============================================================================
def simulate_csr_points(area_dims, density):
    """
    在指定区域内生成符合CSR(泊松点过程)的二维点云。
    """
    area = area_dims[0] * area_dims[1]
    num_points = np.random.poisson(area * density)
    x_coords = np.random.uniform(0, area_dims[0], num_points)
    y_coords = np.random.uniform(0, area_dims[1], num_points)
    print(f"在 {area_dims} 区域内生成了 {num_points} 个点。")
    return np.column_stack((x_coords, y_coords))


def sample_subregion(points, sample_dims, large_area_dims):
    """
    从大型点云中随机采样一个子区域的点。
    """
    max_x_start = large_area_dims[0] - sample_dims[0]
    max_y_start = large_area_dims[1] - sample_dims[1]

    x0 = np.random.uniform(0, max_x_start)
    y0 = np.random.uniform(0, max_y_start)
    x1 = x0 + sample_dims[0]
    y1 = y0 + sample_dims[1]

    mask = (points[:, 0] >= x0) & (points[:, 0] < x1) & \
           (points[:, 1] >= y0) & (points[:, 1] < y1)

    sub_points = points[mask]
    sub_points[:, 0] -= x0
    sub_points[:, 1] -= y0

    # 返回精确的窗口 (0, width, 0, height)
    return sub_points, (0, sample_dims[0], 0, sample_dims[1])


def get_simulation_envelope(sample_points, window_dims, func_name='Lest', n_sim=199, distance_list=None):
    """
    使用spatstat的envelope功能计算置信区间。
    [已修正] 根据 func_name 动态选择 correction
    """
    print(f"正在为 {func_name} 计算理论置信区间...")

    # --- 错误修正 ---
    # 不同的 'fun' (Gest, Fest, Kest) 需要不同的 'correction'
    # 'iso' (isotropic) 适用于 Kest 和 pcf
    # 'km' (Kaplan-Meier) 适用于 Gest, Fest, Jest
    if func_name in ['Kest', 'pcf']:
        correction = "iso"
    elif func_name in ['Gest', 'Fest', 'Jest']:
        correction = "km"  # 与您其他函数保持一致，使用 'km'
    else:
        # 默认为 'iso'，但也可能需要根据新函数进行调整
        print(f"警告: 未知的 func_name '{func_name}'，默认使用 'iso' 修正。")
        correction = "iso"

    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        r.assign("points_x_r", sample_points[:, 0])
        r.assign("points_y_r", sample_points[:, 1])
        r.assign("x_min_r", window_dims[0])
        r.assign("x_max_r", window_dims[1])
        r.assign("y_min_r", window_dims[2])
        r.assign("y_max_r", window_dims[3])
        r.assign("func_r", func_name)
        r.assign("n_sim_r", n_sim)
        r.assign("correction_r", correction)  # 传递修正后的 'correction'

        if distance_list is not None:
            r.assign("distance_list_r", distance_list)
            r_arg = "r=distance_list_r"  # 显式使用您的列表
        else:
            r_arg = "r=NULL"  # 保持 spatstat 默认行为

            # 2. 修改 r_script (Line 825)
        r_script = f"""
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp_obj <- ppp(x=points_x_r, y=points_y_r, window=win)
                # 强制使用恒定强度的CSR进行模拟
                # [已修正] 使用 correction_r 变量，而不是 "iso"
                env <- envelope(ppp_obj, 
                                fun={func_name}, 
                                nsim=n_sim_r, 
                                rank=5, 
                                correction=correction_r, 
                                simulate=expression(rpoispp(ppp_obj$n / area(win), win=win)),
                                {r_arg})  # <--- 在这里添加 r 参数
                env
                """
        env_result = r(r_script)

        r_vals = env_result['r']
        lo_vals = env_result['lo']
        hi_vals = env_result['hi']

    print("置信区间计算完成。")
    return r_vals, lo_vals, hi_vals


# =============================================================================
# 5. 新的绘图功能函数 (根据您的需求定制)
# =============================================================================
def plot_curve_results(distance_list, results, envelope, theoretical_val, title, ylabel, filename):
    """
    绘制曲线特征的模拟结果。

    Args:
        distance_list (array): X轴的距离
        results (list of arrays): 每次模拟的Y轴结果列表
        envelope (tuple): (r_env, lo_env, hi_env) from spatstat
        theoretical_val (float or array): 理论值 (水平线或曲线)
        title (str): 图表标题
        ylabel (str): Y轴标签
        filename (str): 保存的文件名
    """
    plt.figure(figsize=(10, 8))

    # 1. 绘制置信区间
    if envelope:
        r_env, lo_env, hi_env = envelope
        plt.fill_between(r_env, lo_env, hi_env, color='grey', alpha=0.5, label='95% Simulation Envelope')

    # 2. 绘制所有模拟曲线
    for i, res_series in enumerate(results):
        label = 'Simulation Runs' if i == 0 else ''
        plt.plot(distance_list, res_series, color='cornflowerblue', alpha=0.2, label=label)

    # 3. 绘制理论期望线 (支持曲线或水平线)
    if isinstance(theoretical_val, (int, float)):
        plt.axhline(y=theoretical_val, color='red', linestyle='--', linewidth=2,
                    label=f'Theoretical Value = {theoretical_val}')
    else:  # 假设是数组
        plt.plot(distance_list, theoretical_val, color='red', linestyle='--', linewidth=2,
                 label=f'Theoretical Curve')

    plt.title(title, fontsize=16)
    plt.xlabel('Distance (r)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)

    # 合并图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"曲线图已保存至: {filename}")
    plt.close()


def plot_histogram(data, feature_name, theoretical_val=None, filename=None):
    """
    为单个特征绘制直方图。

    Args:
        data (array or list): 1D 数据 (N_SAMPLES 个值)
        feature_name (str): 特征名称 (用于标题)
        theoretical_val (float, optional): 理论值 (绘制垂线)
        filename (str, optional): 保存的文件名
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=30)

    if theoretical_val is not None:
        plt.axvline(x=theoretical_val, color='red', linestyle='--', linewidth=2,
                    label=f'Theoretical Value = {theoretical_val:.4f}')
        plt.legend()

    mean_val = np.nanmean(data)
    std_val = np.nanstd(data)
    plt.title(f'Distribution of {feature_name}\nMean={mean_val:.4f} | StdDev={std_val:.4f}', fontsize=16)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"直方图已保存至: {filename}")
    plt.close()


def plot_combined_violin(data_df, filename):
    """
    为所有单值特征绘制归一化后的组合小提琴图。

    Args:
        data_df (pd.DataFrame): 包含所有单值特征结果的DataFrame
        filename (str): 保存的文件名
    """
    # 1. 归一化 (Z-score)
    data_normalized = data_df.apply(zscore)

    # 2. 将宽数据转换为长数据 (Tidy format) 以便Seaborn使用
    data_melted = data_normalized.melt(var_name='Feature', value_name='Normalized Value (Z-score)')

    # 3. 绘图
    plt.figure(figsize=(15, 8))
    sns.violinplot(x='Feature', y='Normalized Value (Z-score)', data=data_melted, inner='quartile')

    plt.title('Normalized Distribution of All Single-Value Features (Robustness)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle=':', alpha=0.6, axis='y')
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"组合小提琴图已保存至: {filename}")
    plt.close()


# =============================================================================
# 6. 主执行逻辑
# =============================================================================
if __name__ == '__main__':
    # --- 参数配置 ---
    LARGE_AREA_DIMS = (100000, 100000)
    SAMPLE_WINDOW_DIMS = (3000, 3000)
    POINT_DENSITY = 0.0005  # 点密度 (lambda)
    N_SAMPLES = 299
    DISTANCE_LIST = np.linspace(0, 100, 101)  # 用于评估的距离
    SIM_PROCESS_LIST = ["sim_points", "sim_points"]

    # 计算理论lambda值
    LAMBDA_THEO = POINT_DENSITY

    # --- 步骤 1: 生成大型CSR“基底”点云 ---
    print("--- 步骤 1: 生成CSR点云 ---")
    csr_points = simulate_csr_points(LARGE_AREA_DIMS, POINT_DENSITY)

    # --- 步骤 2: 准备结果存储 ---
    print(f"\n--- 步骤 2: 开始 {N_SAMPLES} 次重复采样与计算 ---")

    # 存储曲线特征
    results_curves = {
        'L': [], 'g': [], 'G': [], 'F': [], 'J': [], 'Linhom': [], 'g_inhom': []
    }

    # 存储单值特征 (使用字典，最后转为DataFrame)
    results_single_dict = {
        'ANN_Index': [],
        'Global_Density': [],
        'J_function_Avg': [],
        'density_std_dev': [],
        'density_skewness': [],
        'density_kurtosis': []
    }

    first_sample_points = None
    first_sample_window = None

    for i in tqdm(range(N_SAMPLES), desc="模拟进度"):
        sample_points, window_dims = sample_subregion(csr_points, SAMPLE_WINDOW_DIMS, LARGE_AREA_DIMS)

        if i == 0:
            first_sample_points = sample_points
            first_sample_window = window_dims

        # 准备输入数据
        wsi_id = f'sim_{i}'
        combined_df = pd.DataFrame({
            'wsi_id': wsi_id,
            'topology_x': sample_points[:, 0],
            'topology_y': sample_points[:, 1],
            'object_type': SIM_PROCESS_LIST[0]
        })

        area_df = pd.DataFrame({
            'wsi_id': [wsi_id],
            'tissue_area': [SAMPLE_WINDOW_DIMS[0] * SAMPLE_WINDOW_DIMS[1]],
            'window_dims': [window_dims]  # 传递精确的窗口
        })

        # --- 调用曲线特征函数 ---
        l_result_df = calculate_k_features_r_backend(area_df, combined_df, SIM_PROCESS_LIST, DISTANCE_LIST, if_L=True)
        g_result_df = calculate_g_features_r_backend(area_df, combined_df, SIM_PROCESS_LIST, DISTANCE_LIST)
        gest_result_df = calculate_gest_features_r_backend(area_df, combined_df, SIM_PROCESS_LIST, DISTANCE_LIST)
        fest_result_df = calculate_fest_features_r_backend(area_df, combined_df, SIM_PROCESS_LIST, DISTANCE_LIST)
        jest_result_df = calculate_jest_features_r_backend(area_df, combined_df, SIM_PROCESS_LIST, DISTANCE_LIST)
        linhom_result_df = calculate_Kinhom_features_r_backend(area_df, combined_df, SIM_PROCESS_LIST, DISTANCE_LIST,
                                                               if_L=True)
        ginhom_result_df = calculate_ginhom_features_r_backend(area_df, combined_df, SIM_PROCESS_LIST, DISTANCE_LIST)

        # --- 调用单值特征函数 ---
        ann_index_df = calculate_ann_index_features(combined_df, area_df, [SIM_PROCESS_LIST[0]])
        density_df = calculate_global_density(combined_df, area_df, [SIM_PROCESS_LIST[0]])
        density_stats_df = calculate_density_stats_r_backend(combined_df, [SIM_PROCESS_LIST[0]])

        # --- 存储曲线结果 ---
        if not l_result_df.empty:
            results_curves['L'].append(l_result_df.drop('wsi_id', axis=1).iloc[0].values)
        if not g_result_df.empty:
            results_curves['g'].append(g_result_df.drop('wsi_id', axis=1).iloc[0].values)
        if not gest_result_df.empty:
            results_curves['G'].append(gest_result_df.drop('wsi_id', axis=1).iloc[0].values)
        if not fest_result_df.empty:
            results_curves['F'].append(fest_result_df.drop('wsi_id', axis=1).iloc[0].values)
        if not jest_result_df.empty:
            # 1. 先过滤所有包含 _J_function 的列 (会得到102列)
            all_j_cols = jest_result_df.filter(like='_J_function')
            # 2. 从中移除包含 '_avg_' 的列 (即 _distance_avg_J_function)
            curve_j_cols = all_j_cols.drop(columns=all_j_cols.filter(like='_avg_').columns, errors='ignore')
            # 3. 此时 curve_j_cols 只剩下 101 列，存入结果
            results_curves['J'].append(curve_j_cols.iloc[0].values)
        if not linhom_result_df.empty:
            results_curves['Linhom'].append(linhom_result_df.drop('wsi_id', axis=1).iloc[0].values)
        if not ginhom_result_df.empty:
            results_curves['g_inhom'].append(ginhom_result_df.drop('wsi_id', axis=1).iloc[0].values)

        # --- 存储单值结果 ---
        if not ann_index_df.empty:
            results_single_dict['ANN_Index'].append(ann_index_df.iloc[0][f'ANN_Index_{SIM_PROCESS_LIST[0]}'])
        if not density_df.empty:
            results_single_dict['Global_Density'].append(density_df.iloc[0][f'{SIM_PROCESS_LIST[0]}_global_density'])
        if not jest_result_df.empty:
            results_single_dict['J_function_Avg'].append(
                jest_result_df.iloc[0][f'{SIM_PROCESS_LIST[0]}_{SIM_PROCESS_LIST[1]}_distance_avg_J_function'])
        if not density_stats_df.empty:
            results_single_dict['density_std_dev'].append(
                density_stats_df.iloc[0][f'{SIM_PROCESS_LIST[0]}_density_std_dev'])
            results_single_dict['density_skewness'].append(
                density_stats_df.iloc[0][f'{SIM_PROCESS_LIST[0]}_density_skewness'])
            results_single_dict['density_kurtosis'].append(
                density_stats_df.iloc[0][f'{SIM_PROCESS_LIST[0]}_density_kurtosis'])

    # --- 步骤 3: 转换为DataFrame ---
    df_single_results = pd.DataFrame(results_single_dict)

    # --- 步骤 4: 计算理论置信区间 ---
    print("\n--- 步骤 4: 计算理论置信区间 ---")

    # K-func -> L-func
    k_env = get_simulation_envelope(first_sample_points, first_sample_window, func_name='Kest', n_sim=199,
                                     distance_list=DISTANCE_LIST)
    r_env, lo_k, hi_k = k_env
    lo_l = np.sqrt(np.maximum(0, lo_k) / np.pi) - r_env
    hi_l = np.sqrt(np.maximum(0, hi_k) / np.pi) - r_env
    l_envelope = (r_env, lo_l, hi_l)

    # g-func
    g_envelope = get_simulation_envelope(first_sample_points, first_sample_window, func_name='pcf', n_sim=199,
                                     distance_list=DISTANCE_LIST)
    # G-func
    g_gest_envelope = get_simulation_envelope(first_sample_points, first_sample_window, func_name='Gest', n_sim=199,
                                     distance_list=DISTANCE_LIST)
    # F-func
    f_fest_envelope = get_simulation_envelope(first_sample_points, first_sample_window, func_name='Fest', n_sim=199,
                                     distance_list=DISTANCE_LIST)
    # J-func
    j_jest_envelope = get_simulation_envelope(first_sample_points, first_sample_window, func_name='Jest', n_sim=199,
                                     distance_list=DISTANCE_LIST)

    # (Kinhom 和 pcfinhom 的置信区间较复杂，在此仅与理论值0/1比较)
    linhom_envelope = None
    ginhom_envelope = None

    # --- 步骤 5: 计算非恒定理论曲线 (G 和 F) ---
    theo_G_F_curve = 1 - np.exp(-LAMBDA_THEO * np.pi * DISTANCE_LIST ** 2)

    # --- 步骤 6: 可视化结果 ---
    print("\n--- 步骤 6: 生成结果图表 ---")

    # 绘制曲线图
    plot_curve_results(DISTANCE_LIST, results_curves['L'], l_envelope, 0,
                       'Robustness Test for L-function on CSR Data', 'L(r)', 'robustness_L_function.png')

    plot_curve_results(DISTANCE_LIST, results_curves['g'], g_envelope, 1,
                       'Robustness Test for g-function on CSR Data', 'g(r)', 'robustness_g_function.png')

    plot_curve_results(DISTANCE_LIST, results_curves['G'], g_gest_envelope, theo_G_F_curve,
                       'Robustness Test for G-function on CSR Data', 'G(r)', 'robustness_G_function.png')

    plot_curve_results(DISTANCE_LIST, results_curves['F'], f_fest_envelope, theo_G_F_curve,
                       'Robustness Test for F-function on CSR Data', 'F(r)', 'robustness_F_function.png')

    plot_curve_results(DISTANCE_LIST, results_curves['J'], j_jest_envelope, 1,
                       'Robustness Test for J-function on CSR Data', 'J(r)', 'robustness_J_function.png')

    plot_curve_results(DISTANCE_LIST, results_curves['Linhom'], linhom_envelope, 0,
                       'Robustness Test for Linhom-function on CSR Data', 'L_inhom(r)',
                       'robustness_Linhom_function.png')

    plot_curve_results(DISTANCE_LIST, results_curves['g_inhom'], ginhom_envelope, 1,
                       'Robustness Test for g_inhom-function on CSR Data', 'g_inhom(r)',
                       'robustness_ginhom_function.png')

    # 绘制单值特征的直方图
    plot_histogram(df_single_results['ANN_Index'], 'ANN_Index', 1.0, 'histogram_ANN_Index.png')
    plot_histogram(df_single_results['Global_Density'], 'Global_Density', LAMBDA_THEO, 'histogram_Global_Density.png')
    plot_histogram(df_single_results['J_function_Avg'], 'J_function_Avg (distance_list avg)', 1.0,
                   'histogram_J_function_Avg.png')

    # 理论值未知的直方图
    plot_histogram(df_single_results['density_std_dev'], 'Kernel_Density_StdDev', None, 'histogram_density_std_dev.png')
    plot_histogram(df_single_results['density_skewness'], 'Kernel_Density_Skewness', None,
                   'histogram_density_skewness.png')
    plot_histogram(df_single_results['density_kurtosis'], 'Kernel_Density_Kurtosis', None,
                   'histogram_density_kurtosis.png')

    # 绘制组合小提琴图
    plot_combined_violin(df_single_results, 'robustness_combined_violin_plot.png')

    print("\n模拟与验证完成！所有图表已保存到脚本目录。")