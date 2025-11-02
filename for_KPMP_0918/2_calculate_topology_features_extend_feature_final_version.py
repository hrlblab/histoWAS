from rpy2.robjects import numpy2ri

import os

import pandas as pd

import numpy as np
from scipy.stats import skew, kurtosis

from scipy.spatial.distance import pdist, squareform
import numpy.linalg as linalg

from tqdm import tqdm

import warnings

from sklearn.cluster import DBSCAN

from scipy.spatial import ConvexHull

from skimage.draw import polygon as draw_polygon

from functools import reduce

# 导入rpy2的关键组件

import rpy2.robjects as ro

from rpy2.robjects import r

from rpy2.robjects import pandas2ri

from rpy2.robjects.packages import importr

from rpy2.robjects.conversion import localconverter

# 导入R的基础包和spatstat

base = importr('base')

stats = importr('stats')

spatstat_geom = importr('spatstat.geom')

spatstat_explore = importr('spatstat.explore')

# 忽略Pandas在进行concat操作时可能出现的FutureWarning

warnings.simplefilter(action='ignore', category=FutureWarning)

DBSCAN_EPS = 1000

# min_samples: 一个点被视为核心点所需的邻域内的最小样本数。

# 通常可以从一个较高的值开始尝试，以确保聚类的鲁棒性。

DBSCAN_MIN_SAMPLES = 10

# 定义一个常量，用于centrography计算，一个cluster内至少需要多少个目标点
MIN_POINTS_FOR_CENTROGRAPHY = 10

# 设定密度图的像素大小（以您的坐标单位为准）
# 例如，100意味着每个像素是100x100个单位
DENSITY_MAP_PIXEL_SIZE = 100.0


def _calculate_curve_summary_features(r_distances, r_observed, r_theoretical, feature_prefix, curve_name):
    """
    [新增辅助函数]
    为曲线计算总结性统计特征。
    计算 (观测值 - 理论值) 曲线的 AUC, Max, Min, DistAtMax, DistAtMin。

    Args:
        r_distances (RObject): R返回的距离向量。
        r_observed (RObject): R返回的观测值向量。
        r_theoretical (RObject): R返回的理论值向量。
        feature_prefix (str): 特征名的前缀 (e.g., 'tubules_tubules')。
        curve_name (str): 曲线的名称 (e.g., 'L_function')。

    Returns:
        dict: 包含5个新特征的字典。
    """
    summary_features = {
        f"{feature_prefix}_{curve_name}_AUC": np.nan,
        f"{feature_prefix}_{curve_name}_Max": np.nan,
        f"{feature_prefix}_{curve_name}_DistAtMax": np.nan,
        f"{feature_prefix}_{curve_name}_Min": np.nan,
        f"{feature_prefix}_{curve_name}_DistAtMin": np.nan
    }
    try:
        # --- 计算“中心化”的曲线 ---
        r_centered = np.array(r_observed) - np.array(r_theoretical)
        r_distances_np = np.array(r_distances)

        # 清理数据：移除可能在 r=0 处出现的 NaN 或 Inf
        valid_idx = np.isfinite(r_centered) & np.isfinite(r_distances_np)

        if np.sum(valid_idx) > 1:
            r_dist_valid = r_distances_np[valid_idx]
            r_centered_valid = r_centered[valid_idx]

            # 1. 计算 AUC (使用梯形法则)
            summary_features[f"{feature_prefix}_{curve_name}_AUC"] = np.trapz(r_centered_valid, r_dist_valid)

            # 2. 计算最大偏离
            summary_features[f"{feature_prefix}_{curve_name}_Max"] = np.max(r_centered_valid)

            # 3. 计算最大偏离处的距离
            summary_features[f"{feature_prefix}_{curve_name}_DistAtMax"] = r_dist_valid[np.argmax(r_centered_valid)]

            # 4. 计算最小偏离
            summary_features[f"{feature_prefix}_{curve_name}_Min"] = np.min(r_centered_valid)

            # 5. 计算最小偏离处的距离
            summary_features[f"{feature_prefix}_{curve_name}_DistAtMin"] = r_dist_valid[np.argmin(r_centered_valid)]
    except Exception as e:
        print(f"警告: 为 {feature_prefix}_{curve_name} 计算总结性特征失败。错误: {e}")
        # 特征值默认为NaN，无需额外操作

    return summary_features


def calculate_density_stats_r_backend(combined_df: pd.DataFrame,
                                      process_list: list) -> pd.DataFrame:
    """
    [R后端版 - 已修正rpy2提取错误]
    计算每个WSI中指定对象的核密度图的统计特征。

    使用spatstat的 density.ppp 和 bw.ppl (自动带宽选择)。
    计算密度图像素值的标准差、偏度(skewness)和峰度(kurtosis)。

    Args:
        combined_df (pd.DataFrame): 包含一个病人所有WSI中标注信息的DataFrame.
        process_list (list): 需要计算特征的对象类别, e.g., ['tubules'].

    Returns:
        pd.DataFrame: 每个WSI一行，包含聚合后的密度统计特征。
    """

    results_list = []

    # 1. 按WSI ID分组
    for wsi_id, wsi_df in combined_df.groupby('wsi_id'):
        wsi_result = {'wsi_id': wsi_id}

        # 2. 从当前WSI的所有点中计算实际的边界框 (Bounding Box)
        if wsi_df.empty:
            print(f"警告: WSI {wsi_id} 为空，跳过密度统计。")
            continue

        x_min, x_max = wsi_df['topology_x'].min(), wsi_df['topology_x'].max()
        y_min, y_max = wsi_df['topology_y'].min(), wsi_df['topology_y'].max()

        # 3. 遍历process_list中的每种对象 (e.g., 'tubules')
        for object_type in process_list:

            feature_prefix = f"{object_type}_density"
            default_features = {
                f"{feature_prefix}_std_dev": np.nan,
                f"{feature_prefix}_skewness": np.nan,
                f"{feature_prefix}_kurtosis": np.nan
            }

            # 获取当前WSI中目标对象的点
            points_df = wsi_df[wsi_df['object_type'] == object_type]
            n_points = len(points_df)

            # 需要足够的点来计算密度和统计数据
            if n_points < 10:
                print(f"警告: WSI {wsi_id} 中 '{object_type}' 的点数 ({n_points}) 不足10个，跳过密度统计。")
                wsi_result.update(default_features)
                continue

            # 使用rpy2的上下文管理器
            with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
                try:
                    # 将数据和参数传递给R
                    r.assign("points_x_r", points_df['topology_x'].values)
                    r.assign("points_y_r", points_df['topology_y'].values)
                    r.assign("x_min_r", x_min)
                    r.assign("x_max_r", x_max)
                    r.assign("y_min_r", y_min)
                    r.assign("y_max_r", y_max)
                    r.assign("pixel_size_r", DENSITY_MAP_PIXEL_SIZE)

                    # R 脚本
                    # 1. 创建 ppp 对象
                    # 2. 使用 bw.ppl 自动选择带宽
                    # 3. 计算 density map
                    # 4. 提取像素值
                    r_script = """
                    win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                    ppp_obj <- ppp(x=points_x_r, y=points_y_r, window=win)

                    # 自动选择带宽 (PDF, p. 171)
                    # 使用tryCatch来处理bw.ppl可能失败的情况
                    sigma <- tryCatch({
                      bw.ppl(ppp_obj)
                    }, error = function(e) {
                      # 如果bw.ppl失败, 回退到一个简单的规则 (PDF, p. 171)
                      (x_max_r - x_min_r) / 8 
                    })

                    # 计算密度图 (PDF, p. 168)
                    D <- density(ppp_obj, sigma=sigma, eps=c(pixel_size_r, pixel_size_r))

                    v <- as.vector(D$v)
                    v <- v[is.finite(v)] # 移除 NA 和 Inf
                    v
                    """

                    # 执行 R 脚本并获取像素值向量
                    density_values_r = r(r_script)

                    # --- 关键修改：在Python中进行统计计算 ---
                    # 将R向量转换为Numpy数组
                    density_values_np = np.array(density_values_r)

                    if len(density_values_np) < 4:
                        wsi_result.update(default_features)
                        continue

                    # 使用Numpy和Scipy计算统计数据
                    wsi_result[f"{feature_prefix}_std_dev"] = np.std(density_values_np, ddof=1)
                    # fisher=False 使用与R e1071包、SAS、SPSS相同的G1定义
                    wsi_result[f"{feature_prefix}_skewness"] = skew(density_values_np, bias=False)
                    # fisher=False 使用与SAS、SPSS相同的G2定义（0是正态分布）
                    wsi_result[f"{feature_prefix}_kurtosis"] = kurtosis(density_values_np, bias=False, fisher=True)

                except Exception as e:
                    print(f"处理 WSI {wsi_id} 的 '{object_type}' 时发生R错误: {e}")
                    wsi_result.update(default_features)

        results_list.append(wsi_result)

    return pd.DataFrame(results_list)


# ==============================================================================
# 代码修改区域 1/6: 重构 Gest 函数
# - 移除内部循环，现在只处理单个WSI
# - 接收 distance_values 和 distance_names
# - 返回一个字典而不是DataFrame
# ==============================================================================
def calculate_gest_features_r_backend(wsi_id: str,
                                      wsi_df: pd.DataFrame,
                                      process_list: list,
                                      distance_values: list,
                                      distance_names: list) -> dict:
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    is_univariate = (object_type_1 == object_type_2)
    feature_prefix = f"{object_type_1}_{object_type_2}"
    curve_name = "G_function"

    wsi_result = {'wsi_id': wsi_id}

    # 为所有可能的特征预设NaN
    default_features = _calculate_curve_summary_features([], [], [], feature_prefix, curve_name)
    for name in distance_names:
        default_features[f"{feature_prefix}_distance_{name}_{curve_name}"] = np.nan
    wsi_result.update(default_features)  # 先用NaN填充

    # 使用上下文管理器来激活转换规则
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        try:
            group_df = wsi_df  # 保持内部变量名一致性

            # 1. 检查整个WSI的点数
            if len(group_df) < 2:
                print(f"警告: wsi_id '{wsi_id}' 的总点数 ({len(group_df)}) 不足，已跳过 Gest。")
                return wsi_result

            # 2. 从当前WSI的所有点中计算实际的边界框 (Bounding Box)
            x_min, x_max = group_df['topology_x'].min(), group_df['topology_x'].max()
            y_min, y_max = group_df['topology_y'].min(), group_df['topology_y'].max()

            r.assign("x_min_r", x_min)
            r.assign("x_max_r", x_max)
            r.assign("y_min_r", y_min)
            r.assign("y_max_r", y_max)

            # 3. 检查特定类型的点数
            points1_df = group_df[group_df['object_type'] == object_type_1]
            n1 = len(points1_df)

            if n1 < 1:
                print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_1}' 的点数 ({n1}) 不足，已跳过 Gest。")
                return wsi_result

            r.assign("points1_x_r", points1_df['topology_x'].values)
            r.assign("points1_y_r", points1_df['topology_y'].values)

            if is_univariate:
                if n1 < 2:
                    print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_1}' 的点数 ({n1}) 不足2，跳过单变量G-func。")
                    return wsi_result

                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)
                Gest(ppp1, correction="km") 
                """
                gest_result_r = r(r_script)

            else:  # is_bivariate
                points2_df = group_df[group_df['object_type'] == object_type_2]
                n2 = len(points2_df)
                if n2 < 1:
                    print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_2}' 的点数 ({n2}) 不足，已跳过 Gest。")
                    return wsi_result

                all_x = pd.concat([points1_df['topology_x'], points2_df['topology_x']], ignore_index=True).values
                all_y = pd.concat([points1_df['topology_y'], points2_df['topology_y']], ignore_index=True).values
                all_marks = np.concatenate([np.repeat(object_type_1, n1), np.repeat(object_type_2, n2)])

                r.assign("all_x_r", all_x)
                r.assign("all_y_r", all_y)
                r.assign("all_marks_r", all_marks)
                r.assign("type1_r", object_type_1)
                r.assign("type2_r", object_type_2)

                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                marks_factor <- factor(all_marks_r)
                ppp_marked <- ppp(x=all_x_r, y=all_y_r, window=win, marks=marks_factor)
                Gcross(ppp_marked, i=type1_r, j=type2_r, correction="km")
                """
                gest_result_r = r(r_script)

            summary_feats = _calculate_curve_summary_features(
                gest_result_r['r'], gest_result_r['km'], gest_result_r['theo'],
                feature_prefix, curve_name
            )
            wsi_result.update(summary_feats)

            r_distances = gest_result_r['r']
            r_gest_values = gest_result_r['km']

            gest_values_at_distances = np.interp(distance_values, r_distances, r_gest_values, left=np.nan,
                                                 right=np.nan)
            for i, name in enumerate(distance_names):
                wsi_result[f"{feature_prefix}_distance_{name}_{curve_name}"] = gest_values_at_distances[i]

        except Exception as e:
            print(f"处理 wsi_id '{wsi_id}' 的Gest时发生未知错误: {e}")
            # wsi_result already contains NaNs, so we just return it
    return wsi_result


def _get_mean_std(data_list: list, prefix: str, suffix_mean: str = "_mean", suffix_std: str = "_std"):
    """辅助函数：计算列表的均值和标准差，处理空列表情况"""
    if data_list:
        return {
            f"{prefix}{suffix_mean}": np.mean(data_list),
            f"{prefix}{suffix_std}": np.std(data_list)
        }
    else:
        # 如果列表为空（没有一个cluster达标），返回NaN
        return {
            f"{prefix}{suffix_mean}": np.nan,
            f"{prefix}{suffix_std}": np.nan
        }


def calculate_centrography_features(combined_df: pd.DataFrame,
                                    process_list: list) -> pd.DataFrame:
    """
    计算一个病人文件夹下，每个WSI中每个组织块(DBSCAN cluster)的中心地理学特征,
    并聚合为WSI级别的均值和标准差。

    计算的特征包括:
    1. Mean Center (MC_X, MC_Y): 几何中心
    2. Standard Distance (SD): 离散程度
    3. Standard Deviational Ellipse (SDE): 长轴、短轴、旋转角度

    Args:
        combined_df (pd.DataFrame): 包含一个病人所有WSI中标注信息的DataFrame.
        process_list (list): 需要计算特征的对象类别, e.g., ['tubules'].

    Returns:
        pd.DataFrame: 每个WSI一行，包含聚合后的中心地理学特征。
    """

    results_list = []

    # 1. 按WSI ID分组
    for wsi_id, wsi_df in combined_df.groupby('wsi_id'):
        wsi_result = {'wsi_id': wsi_id}

        # 提取当前WSI的所有点（无论类型），用于DBSCAN地理聚类
        points_all_types = wsi_df[['topology_x', 'topology_y']].values

        if len(points_all_types) < DBSCAN_MIN_SAMPLES:
            print(f"警告: WSI {wsi_id} 点数不足 ({len(points_all_types)})，跳过Centrography计算。")
            # 即使跳过，我们也为process_list中的对象填充NaN
            for object_type in process_list:
                for feature in ["MC_X", "MC_Y", "SD", "SDE_Long", "SDE_Short", "SDE_Angle"]:
                    wsi_result.update(_get_mean_std([], f"{object_type}_{feature}"))
            results_list.append(wsi_result)
            continue

        # 2. 对当前WSI的所有点执行DBSCAN，找到组织块
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1).fit(points_all_types)
        labels = db.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)  # 移除噪声点

        if not unique_labels:
            print(f"警告: WSI {wsi_id} 未能聚类出有效组织块，跳过Centrography计算。")
            for object_type in process_list:
                for feature in ["MC_X", "MC_Y", "SD", "SDE_Long", "SDE_Short", "SDE_Angle"]:
                    wsi_result.update(_get_mean_std([], f"{object_type}_{feature}"))
            results_list.append(wsi_result)
            continue

        # 3. 遍历process_list中的每种对象 (e.g., 'tubules')
        for object_type in process_list:
            # 准备列表来收集WSI内 *每个* cluster的特征值
            cluster_mc_x, cluster_mc_y, cluster_sd = [], [], []
            cluster_sde_long, cluster_sde_short, cluster_sde_angle = [], [], []

            # 获取当前WSI中目标对象的类型掩码
            type_mask = (wsi_df['object_type'].values == object_type)

            # 4. 遍历每个DBSCAN找到的组织块(cluster)
            for label in unique_labels:
                cluster_mask = (labels == label)

                # 找到 *这个cluster* 中 *这个type* 的所有点
                points_for_calc = points_all_types[cluster_mask & type_mask]

                # 检查这个cluster中是否有足够的目标点来进行计算
                if len(points_for_calc) < MIN_POINTS_FOR_CENTROGRAPHY:
                    continue  # 点太少，跳到下一个cluster

                # --- 开始计算特征 ---

                # 1. 平均中心 (Mean Center)
                mc = np.mean(points_for_calc, axis=0)

                cluster_bbox_xmin = np.min(points_for_calc[:, 0])
                cluster_bbox_ymin = np.min(points_for_calc[:, 1])

                cluster_mc_x.append(mc[0] - cluster_bbox_xmin)
                cluster_mc_y.append(mc[1] - cluster_bbox_ymin)

                # 2. 标准距离 (Standard Distance)
                # 计算每个点到平均中心的平方差
                diffs_sq = (points_for_calc - mc) ** 2
                # SD = sqrt( sum( (x-mc_x)^2 + (y-mc_y)^2 ) / N )
                sd = np.sqrt(np.sum(diffs_sq) / len(points_for_calc))
                cluster_sd.append(sd)

                # 3. 标准差椭圆 (SDE)
                # SDE依赖于坐标的协方差矩阵
                # np.cov 期望 (features, observations)，所以我们转置
                cov_matrix = np.cov(points_for_calc.T)

                # 协方差矩阵的特征值和特征向量
                eigenvalues, eigenvectors = linalg.eig(cov_matrix)

                # SDE的轴长正比于特征值的平方根
                # (乘以sqrt(2)以匹配1个标准差的惯例)
                axis_lengths = np.sqrt(2 * eigenvalues)
                long_axis = np.max(axis_lengths)
                short_axis = np.min(axis_lengths)
                cluster_sde_long.append(long_axis)
                cluster_sde_short.append(short_axis)

                # SDE的角度由最大特征值对应的特征向量决定
                long_axis_vector = eigenvectors[:, np.argmax(eigenvalues)]
                # arctan2(y, x) 计算从x轴到(x,y)向量的角度
                angle_rad = np.arctan2(long_axis_vector[1], long_axis_vector[0])
                cluster_sde_angle.append(np.degrees(angle_rad))

            # 5. 聚合(Aggregation): 计算WSI级别上所有cluster特征的均值和标准差
            wsi_result.update(_get_mean_std(cluster_mc_x, f"{object_type}_MC_X"))
            wsi_result.update(_get_mean_std(cluster_mc_y, f"{object_type}_MC_Y"))
            wsi_result.update(_get_mean_std(cluster_sd, f"{object_type}_SD"))
            wsi_result.update(_get_mean_std(cluster_sde_long, f"{object_type}_SDE_Long"))
            wsi_result.update(_get_mean_std(cluster_sde_short, f"{object_type}_SDE_Short"))
            wsi_result.update(_get_mean_std(cluster_sde_angle, f"{object_type}_SDE_Angle"))

        results_list.append(wsi_result)

    return pd.DataFrame(results_list)


# 4. 估算Mask尺寸时，在最大坐标基础上增加的边距（padding）

MASK_PADDING = 200  # 单位与您的坐标系相同


def calculate_global_density(combined_df: pd.DataFrame,
                             area_df: pd.DataFrame,
                             process_list: list) -> pd.DataFrame:
    """
    计算一个病人文件夹下，每个WSI内指定对象的全局平均密度。

    Args:
        combined_df (pd.DataFrame): 包含一个病人所有WSI中标注信息的DataFrame，
                                    应至少包含 'wsi_id' 和 'object_type' 列。
        area_df (pd.DataFrame): 包含每个WSI前景总面积的DataFrame，
                                应包含 'wsi_id' 和 'tissue_area' 列。
        process_list (list): 一个列表，包含需要计算密度的对象类别名称。
                             例如: ['tubules', 'mitotic_figure']

    Returns:
        pd.DataFrame: 一个包含每个WSI及其计算出的密度特征的DataFrame。
                      例如:
                         wsi_id  tubules_density  mitotic_figure_density
                      0  wsi_A.svs         0.001234                0.000567
                      1  wsi_B.svs         0.001198                0.000432
    """
    # 验证输入DataFrame的有效性
    if area_df.empty or 'tissue_area' not in area_df.columns or 'wsi_id' not in area_df.columns:
        print("警告: area_df为空或缺少必要的列 ('wsi_id', 'tissue_area')，无法计算密度。")
        return pd.DataFrame()

    if 'object_type' not in combined_df.columns or 'wsi_id' not in combined_df.columns:
        print("警告: combined_df缺少必要的列 ('wsi_id', 'object_type')，无法计算密度。")
        return pd.DataFrame()

    results_list = []
    # 遍历area_df中的每一行，即每一个WSI
    for index, row in area_df.iterrows():
        wsi_id = row['wsi_id']
        tissue_area = row['tissue_area']

        wsi_result = {'wsi_id': wsi_id}

        # 从大的combined_df中筛选出当前WSI的所有标注
        wsi_annotations = combined_df[combined_df['wsi_id'] == wsi_id]

        # 遍历process_list，为当前WSI计算每种对象的密度
        for object_name in process_list:
            # 计算当前WSI中特定对象的数量
            num_objects = (wsi_annotations['object_type'] == object_name).sum()

            # 计算密度
            feature_name = f"{object_name}_density"
            if tissue_area > 0:
                density = num_objects / tissue_area
            else:
                density = 0.0  # 使用浮点数0.0

            # 将计算结果存入字典
            wsi_result[feature_name] = density

        results_list.append(wsi_result)

    # 将结果列表转换为DataFrame并返回
    if not results_list:
        return pd.DataFrame()

    return pd.DataFrame(results_list)


# ==============================================================================
# 代码修改区域 2/6: 重构 g-function 函数
# - 移除内部循环，现在只处理单个WSI
# - 接收 distance_values 和 distance_names
# - 返回一个字典而不是DataFrame
# ==============================================================================
def calculate_g_features_r_backend(wsi_id: str,
                                   wsi_df: pd.DataFrame,
                                   process_list: list,
                                   distance_values: list,
                                   distance_names: list) -> dict:
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    is_univariate = (object_type_1 == object_type_2)
    feature_prefix = f"{object_type_1}_{object_type_2}"
    curve_name = "g_function"

    wsi_result = {'wsi_id': wsi_id}

    # 为所有可能的特征预设NaN
    default_features = _calculate_curve_summary_features([], [], [], feature_prefix, curve_name)
    for name in distance_names:
        default_features[f"{feature_prefix}_distance_{name}_{curve_name}"] = np.nan
    wsi_result.update(default_features)

    # 使用上下文管理器来激活转换规则
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        try:
            group_df = wsi_df  # 保持内部变量名一致性

            # 1. 检查整个WSI的点数是否过少
            if len(group_df) < 2:
                print(f"警告: wsi_id '{wsi_id}' 的总点数 ({len(group_df)}) 不足，已跳过 g-func。")
                return wsi_result

            # 2. 从当前WSI的所有点中计算实际的边界框 (Bounding Box)
            x_min, x_max = group_df['topology_x'].min(), group_df['topology_x'].max()
            y_min, y_max = group_df['topology_y'].min(), group_df['topology_y'].max()
            r.assign("x_min_r", x_min)
            r.assign("x_max_r", x_max)
            r.assign("y_min_r", y_min)
            r.assign("y_max_r", y_max)

            # 3. 检查特定类型的点数
            points1_df = group_df[group_df['object_type'] == object_type_1]
            n1 = len(points1_df)

            if n1 < 2:
                print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_1}' 的点数 ({n1}) 不足，已跳过 g-func。")
                return wsi_result

            r.assign("points1_x_r", points1_df['topology_x'].values)
            r.assign("points1_y_r", points1_df['topology_y'].values)

            if is_univariate:
                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)
                pcf(ppp1, correction="iso") 
                """
                g_result_r = r(r_script)

            else:  # is_bivariate
                points2_df = group_df[group_df['object_type'] == object_type_2]
                n2 = len(points2_df)
                if n2 < 2:
                    print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_2}' 的点数 ({n2}) 不足，已跳过 g-func。")
                    return wsi_result

                all_x = pd.concat([points1_df['topology_x'], points2_df['topology_x']], ignore_index=True).values
                all_y = pd.concat([points1_df['topology_y'], points2_df['topology_y']], ignore_index=True).values
                all_marks = np.concatenate([np.repeat(object_type_1, n1), np.repeat(object_type_2, n2)])

                r.assign("all_x_r", all_x)
                r.assign("all_y_r", all_y)
                r.assign("all_marks_r", all_marks)
                r.assign("type1_r", object_type_1)
                r.assign("type2_r", object_type_2)

                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                marks_factor <- factor(all_marks_r)
                ppp_marked <- ppp(x=all_x_r, y=all_y_r, window=win, marks=marks_factor)
                pcfcross(ppp_marked, i=type1_r, j=type2_r, correction="iso")
                """
                g_result_r = r(r_script)

            summary_feats = _calculate_curve_summary_features(
                g_result_r['r'], g_result_r['iso'], g_result_r['theo'],
                feature_prefix, curve_name
            )
            wsi_result.update(summary_feats)

            r_distances = g_result_r['r']
            r_g_values = g_result_r['iso']
            g_values_at_distances = np.interp(distance_values, r_distances, r_g_values, left=np.nan, right=np.nan)
            for i, name in enumerate(distance_names):
                wsi_result[f"{feature_prefix}_distance_{name}_{curve_name}"] = g_values_at_distances[i]

        except Exception as e:
            print(f"处理 wsi_id '{wsi_id}' 的g-func时发生未知错误: {e}")

    return wsi_result


# ==============================================================================
# 代码修改区域 3/6: 重构 K/L-function 函数
# - 移除内部循环，现在只处理单个WSI
# - 接收 distance_values 和 distance_names
# - 返回一个字典而不是DataFrame
# ==============================================================================
def calculate_k_features_r_backend(wsi_id: str,
                                   wsi_df: pd.DataFrame,
                                   process_list: list,
                                   distance_values: list,
                                   distance_names: list,
                                   if_L: bool = True) -> dict:
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    is_univariate = (object_type_1 == object_type_2)
    feature_prefix = f"{object_type_1}_{object_type_2}"
    k_curve_name = "K_function"
    l_curve_name = "L_function"

    wsi_result = {'wsi_id': wsi_id}

    # 为所有可能的特征预设NaN
    default_features = _calculate_curve_summary_features([], [], [], feature_prefix, l_curve_name)
    for name in distance_names:
        default_features[f"{feature_prefix}_distance_{name}_{k_curve_name}"] = np.nan
        if if_L:
            default_features[f"{feature_prefix}_distance_{name}_{l_curve_name}"] = np.nan
    wsi_result.update(default_features)

    # 使用上下文管理器来激活转换规则
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        try:
            group_df = wsi_df  # 保持内部变量名一致性

            # 1. 检查整个WSI的点数是否过少
            if len(group_df) < 2:
                print(f"警告: wsi_id '{wsi_id}' 的总点数 ({len(group_df)}) 不足，已跳过 K-func。")
                return wsi_result

            # 2. 从当前WSI的所有点中计算实际的边界框 (Bounding Box)
            x_min, x_max = group_df['topology_x'].min(), group_df['topology_x'].max()
            y_min, y_max = group_df['topology_y'].min(), group_df['topology_y'].max()
            r.assign("x_min_r", x_min)
            r.assign("x_max_r", x_max)
            r.assign("y_min_r", y_min)
            r.assign("y_max_r", y_max)

            # 3. 检查特定类型的点数
            points1_df = group_df[group_df['object_type'] == object_type_1]
            n1 = len(points1_df)

            if n1 < 2:
                print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_1}' 的点数 ({n1}) 不足，已跳过 K-func。")
                return wsi_result

            r.assign("points1_x_r", points1_df['topology_x'].values)
            r.assign("points1_y_r", points1_df['topology_y'].values)

            if is_univariate:
                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)
                Kest(ppp1, correction="iso") 
                """
                k_result_r = r(r_script)
            else:  # is_bivariate
                points2_df = group_df[group_df['object_type'] == object_type_2]
                n2 = len(points2_df)
                if n2 < 2:
                    print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_2}' 的点数 ({n2}) 不足，已跳过 K-func。")
                    return wsi_result

                all_x = pd.concat([points1_df['topology_x'], points2_df['topology_x']], ignore_index=True).values
                all_y = pd.concat([points1_df['topology_y'], points2_df['topology_y']], ignore_index=True).values
                all_marks = np.concatenate([np.repeat(object_type_1, n1), np.repeat(object_type_2, n2)])
                r.assign("all_x_r", all_x)
                r.assign("all_y_r", all_y)
                r.assign("all_marks_r", all_marks)
                r.assign("type1_r", object_type_1)
                r.assign("type2_r", object_type_2)

                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                marks_factor <- factor(all_marks_r)
                ppp_marked <- ppp(x=all_x_r, y=all_y_r, window=win, marks=marks_factor)
                Kcross(ppp_marked, i=type1_r, j=type2_r, correction="iso")
                """
                k_result_r = r(r_script)

            r_distances = np.array(k_result_r['r'])
            r_k_values = np.array(k_result_r['iso'])

            if if_L:
                l_obs_full = np.sqrt(np.maximum(0, r_k_values) / np.pi)
                l_theo_full = r_distances
                summary_feats = _calculate_curve_summary_features(
                    r_distances, l_obs_full, l_theo_full, feature_prefix, l_curve_name
                )
                wsi_result.update(summary_feats)

            k_values_at_distances = np.interp(distance_values, r_distances, r_k_values)
            for i, name in enumerate(distance_names):
                wsi_result[f"{feature_prefix}_distance_{name}_{k_curve_name}"] = k_values_at_distances[i]

            if if_L:
                l_values_at_distances = np.sqrt(np.maximum(0, k_values_at_distances) / np.pi) - np.array(
                    distance_values)
                for i, name in enumerate(distance_names):
                    wsi_result[f"{feature_prefix}_distance_{name}_{l_curve_name}"] = l_values_at_distances[i]
        except Exception as e:
            print(f"处理 wsi_id '{wsi_id}' 的K-func时发生未知错误: {e}")
    return wsi_result


def calculate_k_features(area_df: pd.DataFrame,

                         combined_df: pd.DataFrame,

                         process_list: list,

                         distance_list: list) -> pd.DataFrame:
    """

    计算给定WSI数据中一种或两种对象间的Ripley's K函数特征。


    Args:

        area_df (pd.DataFrame): 包含每个WSI研究区域面积的DataFrame。

                                列: ['wsi_id', 'tissue_area']

        combined_df (pd.DataFrame): 包含所有WSI中所有对象质心坐标的DataFrame。

                                    列: ['wsi_id', 'topology_x', 'topology_y', 'object_type']

        process_list (list): 包含一或两个对象类型字符串的列表。

                             - 单变量分析 (Univariate): ['type_A', 'type_A']

                             - 多变量分析 (Bivariate): ['type_A', 'type_B']

        distance_list (list): 需要计算K函数值的距离d的列表。


    Returns:

        pd.DataFrame: 包含每个WSI及其对应K函数特征的DataFrame。

    """

    # 验证输入

    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]

    object_type_2 = process_list[1]

    is_univariate = (object_type_1 == object_type_2)

    results_list = []

    # 按WSI ID分组进行迭代处理

    for wsi_id, group_df in combined_df.groupby('wsi_id'):

        wsi_result = {'wsi_id': wsi_id}

        try:

            # 1. 获取当前WSI的研究区域面积

            area_info = area_df.loc[area_df['wsi_id'] == wsi_id]

            if area_info.empty:
                print(f"警告: 在 area_df 中未找到 wsi_id '{wsi_id}' 的面积信息，已跳过。")

                continue

            area = area_info['tissue_area'].iloc[0]

            # 2. 提取两种对象的点坐标

            points1_df = group_df[group_df['object_type'] == object_type_1]

            points1 = points1_df[['topology_x', 'topology_y']].values

            if is_univariate:

                points2 = None  # 单变量分析不需要第二组点

            else:

                points2_df = group_df[group_df['object_type'] == object_type_2]

                points2 = points2_df[['topology_x', 'topology_y']].values

            # 3. 检查是否有足够的点进行计算

            if points1.shape[0] < 2 or (not is_univariate and points2.shape[0] < 2):

                print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_1}' 或 '{object_type_2}' 的点数不足，无法计算。")

                # 填充NaN

                for d in distance_list:
                    feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_K_function"

                    wsi_result[feature_name] = np.nan

                results_list.append(wsi_result)

                continue

            # 4. 计算K函数

            # pointpats库会计算一系列距离上的K函数值，我们需要从中插值得到指定距离的值

            if is_univariate:

                # 单变量 K-function

                # k_function = K(points1, area=area)

                k_function = k(points1, area=area)

            else:

                # 交叉 K-function

                # k_function = CrossK(points1, points2, area=area)

                k_function = k(points1, points2=points2, area=area)

            # 5. 使用插值获取指定距离d上的K函数值

            # k_function.support 存储了计算的距离点

            # k_function.k 存储了对应的K函数值

            k_values_at_distances = np.interp(distance_list, k_function.support, k_function.k)

            # 6. 生成特征并存入结果字典

            for i, d in enumerate(distance_list):
                feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_K_function"

                wsi_result[feature_name] = k_values_at_distances[i]

            results_list.append(wsi_result)


        except Exception as e:

            print(f"处理 wsi_id '{wsi_id}' 时发生错误: {e}")

            # 发生任何未知错误时，同样填充NaN

            for d in distance_list:
                feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_K_function"

                wsi_result[feature_name] = np.nan

            results_list.append(wsi_result)

    # 将结果列表转换为DataFrame

    if not results_list:
        return pd.DataFrame()

    return pd.DataFrame(results_list)


def calculate_wsi_foreground_area_fast(wsi_id, wsi_df):
    """

    通过数学计算快速估算单个WSI的前景组织总面积。


    此方法计算每个组织块凸包的面积并直接求和，不生成完整的mask。

    注意：如果不同组织块的凸包有重叠，重叠区域面积会被重复计算。


    Args:

        wsi_id (str): WSI的唯一标识符。

        wsi_df (pd.DataFrame): 包含该WSI所有Object坐标的DataFrame。


    Returns:

        float: WSI的前景总面积（估算值）。如果无法计算则返回0。

    """

    print(f"--- 开始快速计算 WSI ID: {wsi_id} ---")

    # 提取坐标点

    points = wsi_df[['topology_x', 'topology_y']].values

    if len(points) < DBSCAN_MIN_SAMPLES:
        print(f"警告: WSI {wsi_id} 的点数量 ({len(points)}) 过少，已跳过。")

        return

    # 步骤1: 使用DBSCAN进行聚类

    print(f"正在对 {len(points)} 个点进行DBSCAN聚类...")

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1).fit(points)

    labels = db.labels_

    # 获取聚类结果，-1代表噪声点，我们忽略它

    unique_labels = set(labels)

    unique_labels.discard(-1)

    if not unique_labels:
        print(f"警告: WSI {wsi_id} 未能聚类出任何有效的组织块，已跳过。")

        return

    print(f"成功识别出 {len(unique_labels)} 个组织块。")

    total_area = 0.0

    # 遍历每个聚类，计算其凸包面积并累加

    for label in unique_labels:

        cluster_points = points[labels == label]

        if len(cluster_points) < 3:
            continue

        try:

            # 计算凸包

            hull = ConvexHull(cluster_points)

            # hull.volume 在2D情况下就是凸包的面积

            total_area += hull.volume

        except Exception as e:

            print(f"警告: WSI {wsi_id} 的 cluster {label} 计算凸包失败: {e}")

            continue

    print(f"WSI {wsi_id} 的估算总前景面积为: {total_area}")

    return total_area


def generate_mask_for_wsi(wsi_id, wsi_df, output_path):
    """

    为单个WSI生成并保存组织mask。


    Args:

        wsi_id (str): WSI的唯一标识符。

        wsi_df (pd.DataFrame): 包含该WSI所有Object坐标的DataFrame。

        output_path (str): 保存mask的路径。

    """

    print(f"--- 开始处理 WSI ID: {wsi_id} ---")

    # 提取坐标点

    points = wsi_df[['topology_x', 'topology_y']].values

    if len(points) < DBSCAN_MIN_SAMPLES:
        print(f"警告: WSI {wsi_id} 的点数量 ({len(points)}) 过少，已跳过。")

        return

    # 步骤1: 使用DBSCAN进行聚类

    print(f"正在对 {len(points)} 个点进行DBSCAN聚类...")

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1).fit(points)

    labels = db.labels_

    # 获取聚类结果，-1代表噪声点，我们忽略它

    unique_labels = set(labels)

    unique_labels.discard(-1)

    if not unique_labels:
        print(f"警告: WSI {wsi_id} 未能聚类出任何有效的组织块，已跳过。")

        return

    print(f"成功识别出 {len(unique_labels)} 个组织块。")

    # 步骤2: 估算整个WSI的Mask尺寸

    max_x = int(points[:, 0].max()) + MASK_PADDING

    max_y = int(points[:, 1].max()) + MASK_PADDING

    mask_shape = (max_y, max_x)  # (height, width)

    print(f"估算的Mask尺寸 (Height, Width): {mask_shape}")

    # 创建一个空白mask

    mask = np.zeros(mask_shape, dtype=np.uint8)

    # 步骤3: 为每个聚类计算凸包并绘制Mask

    for i, label in enumerate(unique_labels):

        cluster_points = points[labels == label]

        # 凸包至少需要3个点

        if len(cluster_points) < 3:
            continue

        try:

            # 计算凸包

            hull = ConvexHull(cluster_points)

            # 获取凸包的顶点坐标

            hull_vertices = cluster_points[hull.vertices]

            # 在mask上绘制填充后的多边形

            # 注意：skimage.draw.polygon的输入是 (rows, cols)，对应 (y, x)

            rr, cc = draw_polygon(hull_vertices[:, 1], hull_vertices[:, 0], shape=mask.shape)

            # 为每个组织块分配一个唯一的整数ID（1, 2, 3, ...）

            mask[rr, cc] = i + 1


        except Exception as e:

            # Scipy的ConvexHull有时会因为点共线等问题失败

            print(f"警告: WSI {wsi_id} 的 cluster {label} 计算凸包失败: {e}")

            continue

    return mask


def get_tissue_area(combined_df):
    result = []

    # This function is to get the area of the tissue

    full_df = combined_df.copy()

    full_df = full_df[['wsi_id', 'topology_x', 'topology_y']]

    # 按 'wsi_id' 分组，为每个WSI独立生成mask

    for wsi_id, wsi_df in full_df.groupby('wsi_id'):
        area = calculate_wsi_foreground_area_fast(wsi_id, wsi_df)

        result.append({'wsi_id': wsi_id, 'tissue_area': area})

    result_df = pd.DataFrame(result)

    return result_df


# ==============================================================================
# 代码修改区域 4/6: 重构 F-function 函数
# - 移除内部循环，现在只处理单个WSI
# - 接收 distance_values 和 distance_names
# - 返回一个字典而不是DataFrame
# ==============================================================================
def calculate_fest_features_r_backend(wsi_id: str,
                                      wsi_df: pd.DataFrame,
                                      process_list: list,
                                      distance_values: list,
                                      distance_names: list) -> dict:
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    feature_prefix = f"{object_type_1}_{object_type_2}"
    curve_name = "F_function"

    wsi_result = {'wsi_id': wsi_id}

    # 为所有可能的特征预设NaN
    default_features = _calculate_curve_summary_features([], [], [], feature_prefix, curve_name)
    for name in distance_names:
        default_features[f"{feature_prefix}_distance_{name}_{curve_name}"] = np.nan
    wsi_result.update(default_features)

    # 使用上下文管理器来激活转换规则
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        try:
            group_df = wsi_df  # 保持内部变量名一致性

            # 1. 检查整个WSI的点数 (用于定义窗口)
            if len(group_df) < 2:
                print(f"警告: wsi_id '{wsi_id}' 的总点数 ({len(group_df)}) 不足，已跳过 F-func。")
                return wsi_result

            # 2. 从当前WSI的所有点中计算实际的边界框 (Bounding Box)
            x_min, x_max = group_df['topology_x'].min(), group_df['topology_x'].max()
            y_min, y_max = group_df['topology_y'].min(), group_df['topology_y'].max()
            r.assign("x_min_r", x_min)
            r.assign("x_max_r", x_max)
            r.assign("y_min_r", y_min)
            r.assign("y_max_r", y_max)

            # 3. 检查 'to' 类型的点数 (object_type_2)
            points2_df = group_df[group_df['object_type'] == object_type_2]
            n2 = len(points2_df)
            if n2 < 1:
                print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_2}' 的点数 ({n2}) 不足，已跳过 F-func。")
                return wsi_result

            r.assign("points_x_r", points2_df['topology_x'].values)
            r.assign("points_y_r", points2_df['topology_y'].values)

            # F-function 逻辑: 总是计算 F_j(r)
            r_script = """
            win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
            ppp_obj <- ppp(x=points_x_r, y=points_y_r, window=win)
            Fest(ppp_obj, correction="km") 
            """
            fest_result_r = r(r_script)

            summary_feats = _calculate_curve_summary_features(
                fest_result_r['r'], fest_result_r['km'], fest_result_r['theo'],
                feature_prefix, curve_name
            )
            wsi_result.update(summary_feats)

            r_distances = fest_result_r['r']
            r_fest_values = fest_result_r['km']
            fest_values_at_distances = np.interp(distance_values, r_distances, r_fest_values, left=np.nan, right=np.nan)
            for i, name in enumerate(distance_names):
                wsi_result[f"{feature_prefix}_distance_{name}_{curve_name}"] = fest_values_at_distances[i]

        except Exception as e:
            print(f"处理 wsi_id '{wsi_id}' 的F-func时发生未知错误: {e}")

    return wsi_result


# ==============================================================================
# 代码修改区域 5/6: 重构 J-function 函数
# - 移除内部循环，现在只处理单个WSI
# - 接收 distance_values 和 distance_names
# - 返回一个字典而不是DataFrame
# ==============================================================================
def calculate_jest_features_r_backend(wsi_id: str,
                                      wsi_df: pd.DataFrame,
                                      process_list: list,
                                      distance_values: list,
                                      distance_names: list) -> dict:
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    is_univariate = (object_type_1 == object_type_2)
    feature_prefix = f"{object_type_1}_{object_type_2}"
    curve_name = "J_function"

    wsi_result = {'wsi_id': wsi_id}

    # 为所有可能的特征预设NaN
    default_features = _calculate_curve_summary_features([], [], [], feature_prefix, curve_name)
    for name in distance_names:
        default_features[f"{feature_prefix}_distance_{name}_{curve_name}"] = np.nan
    default_features[f"{feature_prefix}_distance_avg_{curve_name}"] = np.nan
    wsi_result.update(default_features)

    # 使用上下文管理器来激活转换规则
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        try:
            group_df = wsi_df  # 保持内部变量名一致性

            # 1. 检查整个WSI的点数
            if len(group_df) < 2:
                print(f"警告: wsi_id '{wsi_id}' 的总点数 ({len(group_df)}) 不足，已跳过 J-func。")
                return wsi_result

            # 2. 从当前WSI的所有点中计算实际的边界框 (Bounding Box)
            x_min, x_max = group_df['topology_x'].min(), group_df['topology_x'].max()
            y_min, y_max = group_df['topology_y'].min(), group_df['topology_y'].max()
            r.assign("x_min_r", x_min)
            r.assign("x_max_r", x_max)
            r.assign("y_min_r", y_min)
            r.assign("y_max_r", y_max)

            # 3. 检查特定类型的点数
            points1_df = group_df[group_df['object_type'] == object_type_1]
            n1 = len(points1_df)

            if is_univariate:
                if n1 < 2:
                    print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_1}' 的点数 ({n1}) 不足2，跳过单变量J-func。")
                    return wsi_result

                r.assign("points1_x_r", points1_df['topology_x'].values)
                r.assign("points1_y_r", points1_df['topology_y'].values)

                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)
                Jest(ppp1, correction="km") 
                """
                jest_result_r = r(r_script)

            else:  # is_bivariate
                points2_df = group_df[group_df['object_type'] == object_type_2]
                n2 = len(points2_df)
                if n1 < 1 or n2 < 1:
                    print(
                        f"警告: wsi_id '{wsi_id}' 中 '{object_type_1}'({n1}) 或 '{object_type_2}'({n2}) 点数不足，跳过双变量J-func。")
                    return wsi_result

                all_x = pd.concat([points1_df['topology_x'], points2_df['topology_x']], ignore_index=True).values
                all_y = pd.concat([points1_df['topology_y'], points2_df['topology_y']], ignore_index=True).values
                all_marks = np.concatenate([np.repeat(object_type_1, n1), np.repeat(object_type_2, n2)])

                r.assign("all_x_r", all_x)
                r.assign("all_y_r", all_y)
                r.assign("all_marks_r", all_marks)
                r.assign("type1_r", object_type_1)
                r.assign("type2_r", object_type_2)

                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                marks_factor <- factor(all_marks_r)
                ppp_marked <- ppp(x=all_x_r, y=all_y_r, window=win, marks=marks_factor)
                Jcross(ppp_marked, i=type1_r, j=type2_r, correction="km")
                """
                jest_result_r = r(r_script)

            summary_feats = _calculate_curve_summary_features(
                jest_result_r['r'], jest_result_r['km'], jest_result_r['theo'],
                feature_prefix, curve_name
            )
            wsi_result.update(summary_feats)

            r_distances = jest_result_r['r']
            r_jest_values = jest_result_r['km']
            jest_values_at_distances = np.interp(distance_values, r_distances, r_jest_values, left=np.nan, right=np.nan)

            for i, name in enumerate(distance_names):
                wsi_result[f"{feature_prefix}_distance_{name}_{curve_name}"] = jest_values_at_distances[i]

            wsi_result[f"{feature_prefix}_distance_avg_{curve_name}"] = np.nanmean(jest_values_at_distances)

        except Exception as e:
            print(f"处理 wsi_id '{wsi_id}' 的J-func时发生未知错误: {e}")

    return wsi_result


def _calculate_ann(points):
    """

    为一个坐标点列表计算平均最近邻距离 (ANN)。


    参数:

    points (np.ndarray): 一个N x 2的NumPy数组，包含(x, y)坐标。


    返回:

    float: 计算出的ANN值。如果点的数量少于2，返回np.nan。

    """

    # 至少需要两个点才能计算距离

    if points.shape[0] < 2:
        return np.nan

    # 使用scipy计算所有点对之间的欧氏距离，生成一个压缩的距离矩阵

    distance_matrix = pdist(points, 'euclidean')

    # 将压缩的距离矩阵转换为方形矩阵

    square_distance_matrix = squareform(distance_matrix)

    # 将对角线（点到自身的距离，为0）替换为无穷大，以便在查找最小值时不被选中

    np.fill_diagonal(square_distance_matrix, np.inf)

    # 对于每个点，找到其最近邻的距离（即每行/列的最小值）

    # axis=1表示我们沿着每一行查找最小值

    nearest_neighbor_distances = np.min(square_distance_matrix, axis=1)

    # ANN是所有最近邻距离的平均值

    ann_value = np.mean(nearest_neighbor_distances)

    return ann_value


def calculate_ann_index_features(combined_df: pd.DataFrame,
                                 area_df: pd.DataFrame,
                                 process_list: list) -> pd.DataFrame:
    """
    计算一个WSI内，指定对象的平均最近邻指数 (ANN Index / Clark-Evans Index)。
    该指数通过观测到的ANN与CSR下的预期ANN的比率，对密度进行了归一化。


    Args:
        combined_df (pd.DataFrame): 包含一个病人所有WSI中标注信息的DataFrame。
        area_df (pd.DataFrame): 包含每个WSI前景总面积的DataFrame。
        process_list (list): 需要计算特征的对象类别, e.g., ['tubules'].

    Returns:
        pd.DataFrame: 每个WSI一行，包含ANN Index特征。
    """

    results_list = []

    # 按 'wsi_id' 分组
    grouped = combined_df.groupby('wsi_id')

    for wsi_id, group_df in grouped:
        wsi_result = {'wsi_id': wsi_id}

        # 1. 获取当前WSI的面积信息
        try:
            tissue_area = area_df.loc[area_df['wsi_id'] == wsi_id, 'tissue_area'].iloc[0]
        except IndexError:
            print(f"警告: WSI {wsi_id} 在 area_df 中无面积信息，跳过ANN Index计算。")
            tissue_area = 0

        # 2. 为这个WSI计算每种指定对象的ANN Index
        for obj_type in process_list:
            feature_name = f"ANN_Index_{obj_type}"

            # 提取这个WSI上特定对象类型的坐标点
            points_df = group_df[group_df['object_type'] == obj_type]
            points = points_df[['topology_x', 'topology_y']].values
            n_points = len(points)

            # ANN Index 需要至少2个点和正面积
            if n_points < 2 or tissue_area <= 0:
                wsi_result[feature_name] = np.nan
                continue

            # 3. 计算观测值 (Observed ANN)
            # (调用您现有的辅助函数)
            observed_ann = _calculate_ann(points)

            # 4. 计算预期值 (Expected ANN for CSR)
            # 密度 (lambda)
            lambda_density = n_points / tissue_area

            # 预期最近邻距离
            expected_ann = 1.0 / (2.0 * np.sqrt(lambda_density))

            # 5. 计算 Clark-Evans Index
            if expected_ann > 0:
                ann_index = observed_ann / expected_ann
            else:
                ann_index = np.nan

            wsi_result[feature_name] = ann_index

        results_list.append(wsi_result)

    if not results_list:
        return pd.DataFrame()

    return pd.DataFrame(results_list)


def calculate_ann_features(grouped, object_types_to_analyze, wsi_ids):
    """

    遍历指定的文件夹结构，读取CSV文件，并为每个WSI计算指定对象的ANN特征。


    参数:

    input_path (str): 包含所有病患文件夹的根目录路径。

    object_types_to_analyze (list): 一个字符串列表，指定要分析的对象类型。

                                     例如: ['arteries', 'tubules']


    返回:

    pd.DataFrame: 一个DataFrame，其中每行代表一个wsi_id，

                  每列是该WSI上对应对象的ANN特征值。

    """

    results = []

    for wsi_id in tqdm(wsi_ids, desc="计算各WSI的ANN"):

        # 为每个WSI创建一个结果字典

        wsi_result = {'wsi_id': wsi_id}

        # 为这个WSI计算每种指定对象的ANN

        for obj_type in object_types_to_analyze:

            try:

                # 提取这个WSI上特定对象类型的坐标点

                group = grouped.get_group((wsi_id, obj_type))

                points = group[['topology_x', 'topology_y']].values

            except KeyError:

                # 如果在这个WSI上找不到这种对象类型的数据，则其ANN值为NaN

                points = np.empty((0, 2))

            # 计算ANN值

            ann_value = _calculate_ann(points)

            # 将结果存入字典，列名为 'ANN_对象类型'

            wsi_result[f'ANN_{obj_type}'] = ann_value

        results.append(wsi_result)

    print("\nANN 计算完成！")

    # 将结果列表转换为DataFrame并返回

    final_df = pd.DataFrame(results)

    return final_df


if __name__ == '__main__':

    input_data_path = "/media/yangy50/Elements/KPMP_new/concate_csv_topology/"
    result_path = "/media/yangy50/Elements/KPMP_new/topology_feature.csv"

    filename_map = {

        # 'arteries': 'arteriesarterioles_Features.csv',
        #
        # 'globally_sclerotic_glomeruli': 'globally_sclerotic_glomeruli_Features.csv',
        #
        # 'non_globally_sclerotic_glomeruli': 'non_globally_sclerotic_glomeruli_Features.csv',

        'tubules': 'tubules_Features.csv'

    }

    objects_to_process = ['tubules']
    # objects_to_process = ['arteries', 'tubules', 'globally_sclerotic_glomeruli', 'non_globally_sclerotic_glomeruli']

    all_patients_data = []

    for patient_folder in tqdm(os.listdir(input_data_path), desc="处理病患文件夹"):

        patient_folder_path = os.path.join(input_data_path, patient_folder)

        if not os.path.isdir(patient_folder_path):
            continue

        all_data = []
        for obj_type in objects_to_process:
            csv_filename = filename_map[obj_type]
            csv_path = os.path.join(patient_folder_path, csv_filename)
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df['object_type'] = obj_type
                    all_data.append(df)
                except Exception as e:
                    print(f"错误: 无法读取或处理文件 {csv_path}。错误信息: {e}")

        if not all_data:
            print(f"警告: 在病人 {patient_folder} 文件夹中未找到任何有效的数据文件，已跳过。")
            continue

        combined_df = pd.concat(all_data, ignore_index=True)
        required_cols = ['wsi_id', 'topology_x', 'topology_y', 'object_type']
        combined_df = combined_df[required_cols]

        # ==============================================================================
        # 代码修改区域 6/6: 重构主执行逻辑
        # 1. 首先计算所有不依赖动态距离的 "批处理" 特征
        # ==============================================================================
        print(f"\n--- 病人 {patient_folder}: 开始计算批处理特征 ---")
        grouped = combined_df.groupby(['wsi_id', 'object_type'])
        wsi_ids = combined_df['wsi_id'].unique()

        # 计算 ANN (这是后续计算的基础)
        ann_feature_df = calculate_ann_features(grouped, objects_to_process, wsi_ids)
        # 计算面积
        area_df = get_tissue_area(combined_df)
        # 计算 ANN 指数
        ann_index_feature_df = calculate_ann_index_features(combined_df, area_df, objects_to_process)
        # 计算全局密度
        density_feature_df = calculate_global_density(combined_df, area_df, objects_to_process)
        # 计算中心地理学特征
        centrography_feature_df = calculate_centrography_features(combined_df, objects_to_process)
        # 计算核密度统计
        # density_stats_df = calculate_density_stats_r_backend(combined_df, objects_to_process)
        print(f"--- 病人 {patient_folder}: 批处理特征计算完成 ---")

        # ==============================================================================
        # 2. 新增WSI级别的循环，以计算依赖于每个WSI自身ANN值的特征
        # ==============================================================================
        spatial_stats_results_list = []
        distance_multipliers = [1, 2, 3, 4]
        distance_names = ['ANN', '2ANN', '3ANN', '4ANN']

        for wsi_id in tqdm(wsi_ids, desc=f"处理病人 {patient_folder} 的空间统计"):
            wsi_level_features = {'wsi_id': wsi_id}

            # --- 为当前WSI获取必要信息 ---
            # 假设只处理tubules
            ann_col_name = f'ANN_{objects_to_process[0]}'
            try:
                ann_value = ann_feature_df.loc[ann_feature_df['wsi_id'] == wsi_id, ann_col_name].iloc[0]
            except (IndexError, KeyError):
                ann_value = np.nan

            wsi_df = combined_df[combined_df['wsi_id'] == wsi_id]

            # 检查是否可以进行计算 (需要有效的ANN值)
            if pd.isna(ann_value) or ann_value <= 0:
                print(f"警告: WSI {wsi_id} 的ANN值无效 ({ann_value})，跳过空间统计计算。")
                spatial_stats_results_list.append(wsi_level_features)
                continue

            # --- 创建动态距离列表 ---
            distance_values = [m * ann_value for m in distance_multipliers]

            # --- 调用修改后的函数 (现在它们一次只处理一个WSI) ---
            process_pair = ["tubules", "tubules"]
            k_features = calculate_k_features_r_backend(wsi_id, wsi_df, process_pair, distance_values, distance_names)
            g_features = calculate_g_features_r_backend(wsi_id, wsi_df, process_pair, distance_values, distance_names)
            gest_features = calculate_gest_features_r_backend(wsi_id, wsi_df, process_pair, distance_values,
                                                              distance_names)
            fest_features = calculate_fest_features_r_backend(wsi_id, wsi_df, process_pair, distance_values,
                                                              distance_names)
            jest_features = calculate_jest_features_r_backend(wsi_id, wsi_df, process_pair, distance_values,
                                                              distance_names)

            # --- 合并当前WSI的所有空间统计特征 ---
            wsi_level_features.update(k_features)
            wsi_level_features.update(g_features)
            wsi_level_features.update(gest_features)
            wsi_level_features.update(fest_features)
            wsi_level_features.update(jest_features)

            spatial_stats_results_list.append(wsi_level_features)

        # 将WSI级别的结果列表转换为DataFrame
        spatial_stats_df = pd.DataFrame(spatial_stats_results_list)

        # ==============================================================================
        # 3. 合并所有特征DataFrame
        # ==============================================================================
        features_to_merge = [
            ann_index_feature_df,
            ann_feature_df,
            density_feature_df,
            centrography_feature_df,
            # density_stats_df,
            spatial_stats_df  # 这是新生成的包含所有空间统计的DataFrame
        ]

        # 过滤掉空的DataFrame
        features_to_merge = [df for df in features_to_merge if not df.empty]

        if not features_to_merge:
            print(f"警告: 病人 {patient_folder} 没有计算出任何有效的特征，已跳过。")
            continue

        # 使用 reduce 和 pd.merge 将列表中的所有DataFrame合并
        patient_merged_df = reduce(lambda left, right: pd.merge(left, right, on='wsi_id', how='outer'),
                                   features_to_merge)

        patient_merged_df['Biopsy ID: '] = patient_folder
        all_patients_data.append(patient_merged_df)
        print(f"--- 病人 {patient_folder} 处理完成 ---")

    # ==============================================================================
    # 4. 最后的数据聚合与保存
    # ==============================================================================
    if all_patients_data:
        final_topology_df = pd.concat(all_patients_data, ignore_index=True)

        final_topology_df.drop(columns=['wsi_id'], axis=1, inplace=True)
        ave_result = final_topology_df.groupby('Biopsy ID: ').mean()

        final_topology_df.to_csv(result_path, index=False)

        print("\n=======================================================")
        print(f"所有计算完成！最终结果已保存至: {result_path}")
        print("最终DataFrame的前几行:")
        print(final_topology_df.head())
        print("=======================================================")
    else:
        print("\n--- 未处理任何数据，无法生成最终的 topology.csv 文件。---")