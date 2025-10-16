from rpy2.robjects import numpy2ri

import os

import pandas as pd

import numpy as np

import math

from scipy.spatial.distance import pdist, squareform

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
from rpy2.rinterface import NULL as R_NULL


# 导入R的基础包和spatstat

base = importr('base')

stats = importr('stats')

spatstat_geom = importr('spatstat.geom')

spatstat_explore = importr('spatstat.explore')
spatstat_model = importr('spatstat.model')


DBSCAN_EPS = 3000
MASK_PADDING = 200
DBSCAN_MIN_SAMPLES = 5


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

    result=[]

    # This function is to get the area of the tissue

    full_df=combined_df.copy()

    full_df=full_df[['wsi_id', 'topology_x', 'topology_y']]

    # 按 'wsi_id' 分组，为每个WSI独立生成mask

    for wsi_id, wsi_df in full_df.groupby('wsi_id'):

        area=calculate_wsi_foreground_area_fast(wsi_id, wsi_df)

        result.append({'wsi_id': wsi_id, 'tissue_area': area})


    result_df = pd.DataFrame(result)


    return result_df


def _apply_dbscan_clustering(wsi_df: pd.DataFrame, wsi_id: str):
    """
    对单个WSI的所有点应用DBSCAN聚类。

    Args:
        wsi_df (pd.DataFrame): 包含该WSI所有对象坐标的DataFrame。
        wsi_id (str): WSI的唯一标识符。

    Returns:
        pd.DataFrame: 包含 'topology_x', 'topology_y', 'cluster_id' 的DataFrame。
    """
    points = wsi_df[['topology_x', 'topology_y']].values

    # 点太少，无法进行有意义的聚类
    if len(points) < DBSCAN_MIN_SAMPLES:
        return pd.DataFrame()

    try:
        # 使用文件顶部的全局常量进行聚类
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1).fit(points)

        # 将聚类标签 (-1 为噪声点) 赋给新列
        result_df = wsi_df[['topology_x', 'topology_y']].copy()
        result_df['cluster_id'] = db.labels_.astype(str)  # 转换为字符串，以处理 -1 标签
        result_df['wsi_id'] = wsi_id

        return result_df

    except Exception as e:
        print(f"警告: WSI {wsi_id} DBSCAN 聚类失败: {e}")
        return pd.DataFrame()


def get_tissue_area_and_clusters(combined_df):
    """
    [修改后的版本] 计算WSI前景组织总面积，并返回带有聚类ID的完整点列表。

    Returns:
        tuple: (area_df, all_clustered_points_df)
    """
    area_results = []
    all_clustered_points = []
    all_wsi_hulls = {}

    # 按 'wsi_id' 分组，为每个WSI独立处理
    for wsi_id, wsi_df in combined_df.groupby('wsi_id'):

        # 1. 应用聚类
        clustered_df = _apply_dbscan_clustering(wsi_df.copy(), wsi_id)

        if clustered_df.empty:
            area_results.append({'wsi_id': wsi_id, 'tissue_area': 0.0})
            all_wsi_hulls[wsi_id] = []
            continue

        # 2. 从聚类结果中计算面积 (只考虑非噪声点)
        non_noise_points = clustered_df[clustered_df['cluster_id'] != '-1']

        # 提取聚类信息用于计算面积
        temp_df = wsi_df.copy()
        temp_df['cluster_id'] = clustered_df['cluster_id']

        area = 0.0
        hull_vertices_list = []
        # 遍历每个非噪声聚类，计算凸包面积
        for label in temp_df['cluster_id'].unique():
            if label == '-1':
                continue

            cluster_points = temp_df[temp_df['cluster_id'] == label][['topology_x', 'topology_y']].values

            if len(cluster_points) < 3:
                continue

            try:
                # 计算凸包
                hull = ConvexHull(cluster_points)
                area += hull.volume

                hull_vertices = cluster_points[hull.vertices]
                hull_vertices_list.append(hull_vertices)

            except Exception:
                continue

        area_results.append({'wsi_id': wsi_id, 'tissue_area': area})
        all_wsi_hulls[wsi_id] = hull_vertices_list

        # 3. 收集带有聚类ID的完整点列表
        # 重新合并 object_type, 但只保留非噪声点
        merged_clustered_df = pd.merge(wsi_df, clustered_df[['wsi_id', 'topology_x', 'topology_y', 'cluster_id']],
                                       on=['wsi_id', 'topology_x', 'topology_y'], how='left')

        # 将噪声点(-1)从列表中移除，因为我们只关心组织块的特征
        merged_clustered_df = merged_clustered_df[merged_clustered_df['cluster_id'] != '-1'].reset_index(drop=True)

        all_clustered_points.append(merged_clustered_df)

    area_df = pd.DataFrame(area_results)
    all_clustered_points_df = pd.concat(all_clustered_points,
                                        ignore_index=True) if all_clustered_points else pd.DataFrame()

    return area_df, all_clustered_points_df,all_wsi_hulls


def calculate_centrography_features(clustered_df: pd.DataFrame,
                                    process_list: list) -> pd.DataFrame:
    """
    [最终长格式版] 计算每个WSI中，每个DBSCAN Cluster的质心分析特征。
    返回长格式 DataFrame，每行代表一个 Cluster，特征列名不带 Cluster ID 后缀。

    Args:
        clustered_df (pd.DataFrame): 包含聚类ID的点数据。
                                     列: ['wsi_id', 'topology_x', 'topology_y', 'object_type', 'cluster_id']
        process_list (list): 包含要分析的对象类型列表。

    Returns:
        pd.DataFrame: 长格式 DataFrame，包含 'wsi_id', 'cluster_id', 'object_type' 和质心特征。
    """
    if clustered_df.empty:
        return pd.DataFrame()

    # 1. 只筛选 process_list 中的对象类型
    target_df = clustered_df[clustered_df['object_type'].isin(process_list)].copy()

    if target_df.empty:
        return pd.DataFrame()

    results_list = []

    # 按照 WSI ID, Cluster ID 和 Object Type 分组
    for (wsi_id, cluster_id, object_type), group in target_df.groupby(['wsi_id', 'cluster_id', 'object_type']):

        # 确保每个组有足够的点来计算 (至少3个点)
        if len(group) < 3:
            # 对于点数不足的 Cluster，我们仍将其记录为 NaN，以便在最终表格中有所体现
            results_list.append({
                'wsi_id': wsi_id,
                'cluster_id': cluster_id,
                'object_type': object_type,
                'MC_X': np.nan,
                'MC_Y': np.nan,
                'SD': np.nan,
                'SDE_Major': np.nan,
                'SDE_Minor': np.nan,
                'SDE_Rotation_Rad': np.nan,
            })
            continue

        n = len(group)
        points = group[['topology_x', 'topology_y']].values

        # Mean Center (mu_x, mu_y)
        mu_x = points[:, 0].mean()
        mu_y = points[:, 1].mean()
        x_dev = points[:, 0] - mu_x
        y_dev = points[:, 1] - mu_y

        # Standard Distance (标准距离)
        sd = np.sqrt(np.sum(x_dev ** 2 + y_dev ** 2) / n)

        # SDE 计算
        sum_x_dev_sq = np.sum(x_dev ** 2)
        sum_y_dev_sq = np.sum(y_dev ** 2)
        sum_xy_dev = np.sum(x_dev * y_dev)

        # 计算旋转角度 (Theta) - 保持不变
        numerator = sum_x_dev_sq - sum_y_dev_sq
        denominator = 2 * sum_xy_dev

        if denominator == 0:
            theta = 0.0 if numerator >= 0 else math.pi / 2.0
        else:
            theta = 0.5 * np.arctan(denominator / numerator)
            if numerator < 0:
                theta += math.pi / 2.0

        # SDE 长短轴 (Major/Minor Axes Standard Deviation)
        sd_major_sq = (numerator / 2.0 + np.sqrt(numerator ** 2 / 4.0 + sum_xy_dev ** 2)) / n
        sd_minor_sq = (sum_x_dev_sq + sum_y_dev_sq) / n - sd_major_sq

        sd_major = np.sqrt(np.maximum(0, sd_major_sq))
        sd_minor = np.sqrt(np.maximum(0, sd_minor_sq))

        # --- 结果字典 (长格式，无后缀) ---
        results_list.append({
            'wsi_id': wsi_id,
            'cluster_id': cluster_id,
            'object_type': object_type,
            'MC_X': mu_x,
            'MC_Y': mu_y,
            'SD': sd,
            'SDE_Major': sd_major,
            'SDE_Minor': sd_minor,
            'SDE_Rotation_Rad': theta,
        })

    return pd.DataFrame(results_list)

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



def calculate_ann_features(grouped, object_types_to_analyze,wsi_ids):

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


    print("\n所有计算完成！")


    # 将结果列表转换为DataFrame并返回

    final_df = pd.DataFrame(results)

    return final_df


def calculate_global_density(combined_df: pd.DataFrame,
                             area_df: pd.DataFrame,
                             process_list: list) -> pd.DataFrame:
    """
    计算每个WSI中特定对象点图案的全局密度 (n/A)。

    Args:
        combined_df (pd.DataFrame): 包含所有对象点坐标的DataFrame。
                                    列: ['wsi_id', 'topology_x', 'topology_y', 'object_type']
        area_df (pd.DataFrame): 包含每个WSI研究区域面积的DataFrame。
                                列: ['wsi_id', 'tissue_area']
        process_list (list): 包含要分析的至少一个对象类型字符串的列表。
                             函数将为列表中的每个唯一对象类型计算密度。

    Returns:
        pd.DataFrame: 包含每个WSI及其对应全局密度特征的DataFrame。
    """
    if not isinstance(process_list, list) or len(process_list) < 1:
        raise ValueError("process_list必须是一个包含至少一个对象类型字符串的列表。")

    results_list = []

    # 遍历 process_list 中所有唯一的对象类型
    for object_type in set(process_list):

        # 1. 筛选出目标对象类型的数据
        # 使用 .copy() 避免 SettingWithCopyWarning
        target_df = combined_df[combined_df['object_type'] == object_type].copy()

        if target_df.empty:
            continue

        # 2. 计算每个WSI中目标对象的总点数 (n)
        n_df = target_df.groupby('wsi_id').size().reset_index(name='n_points')

        # 3. 合并点数和面积信息
        density_df = pd.merge(n_df, area_df, on='wsi_id', how='left')

        # 4. 计算全局密度: density = n / A
        # 确保 tissue_area > 0 且非 NaN
        density_df['global_density'] = np.where(
            (density_df['tissue_area'] > 0) & density_df['tissue_area'].notna(),
            density_df['n_points'] / density_df['tissue_area'],
            np.nan
        )

        # 5. 筛选结果并重命名列
        feature_name = f'Global_Density_{object_type}'
        result_df = density_df[['wsi_id', 'global_density']].rename(
            columns={'global_density': feature_name}
        )

        results_list.append(result_df)

    if not results_list:
        # 如果没有可处理的对象类型，返回一个包含所有WSI ID的空特征DataFrame
        return pd.DataFrame({'wsi_id': combined_df['wsi_id'].unique()})

    # 使用 reduce 和 pd.merge 将所有结果 DataFrame 合并
    final_density_df = reduce(lambda left, right: pd.merge(left, right, on='wsi_id', how='outer'),
                              results_list)

    # 确保包含所有WSI ID（处理某些WSI缺少特定对象类型点数的情况）
    all_wsi_ids = combined_df['wsi_id'].unique()
    final_result_df = pd.DataFrame({'wsi_id': all_wsi_ids}).merge(final_density_df, on='wsi_id', how='left')

    return final_result_df


def calculate_ppm_features_r_backend(combined_df: pd.DataFrame,
                                     area_df: pd.DataFrame,
                                     all_wsi_hulls: dict,
                                     process_list: list,
                                     covariate_column: str,
                                     if_plot: bool = False) -> pd.DataFrame:
    """
    [R后端版] 使用 rpy2 调用 R 的 spatstat 包计算 Poisson Point Process (PPM) 特征。
    将点级协变量插值到组织 Mask 区域，并拟合 PPM 模型。

    Args:
        combined_df (pd.DataFrame): 包含所有对象点坐标和点级协变量的DataFrame。
                                    列: ['wsi_id', 'topology_x', 'topology_y', 'object_type', covariate_column, ...]
        area_df (pd.DataFrame): 包含每个WSI研究区域面积的DataFrame。
        process_list (list): 包含要作为 PPM 目标点模式分析的一个对象类型（如 ['tubules']）。
        covariate_column (str): 要作为 PPM 协变量的列名。
        if_plot (bool): 是否将协变量强度表面可视化 (R代码，仅用于诊断/美观)。

    Returns:
        pd.DataFrame: 包含每个WSI PPM 特征 (Beta, P-value, LogLikelihood) 的 DataFrame。
    """
    if not isinstance(process_list, list) or len(process_list) != 1:
        raise ValueError("process_list 必须是一个包含单一对象类型字符串的列表。")

    object_type = process_list[0]

    # PPM 结果列表
    results_list = []

    # 使用上下文管理器来激活转换规则
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        for wsi_id, group_df in combined_df.groupby('wsi_id'):
            wsi_result = {'wsi_id': wsi_id}

            # 1. 准备数据
            points_df = group_df[group_df['object_type'] == object_type].copy()

            # 确保协变量列存在
            if covariate_column not in group_df.columns:
                print(f"警告: 协变量列 '{covariate_column}' 在 wsi_id '{wsi_id}' 中不存在。")
                for feature in ['Beta_Z', 'P_Value_Z', 'LogLikelihood']:
                    wsi_result[f'PPM_{feature}_{object_type}_{covariate_column}'] = np.nan
                results_list.append(wsi_result)
                continue

            # 提取所有点的坐标和协变量 Z
            all_points = group_df[['topology_x', 'topology_y', covariate_column]].values
            target_points = points_df[['topology_x', 'topology_y']].values

            n_target = len(target_points)
            if n_target < 5:
                print(f"警告: wsi_id '{wsi_id}' 中 '{object_type}' 点数 ({n_target}) 过少，已跳过 PPM。")
                for feature in ['Beta_Z', 'P_Value_Z', 'LogLikelihood']:
                    wsi_result[f'PPM_{feature}_{object_type}_{covariate_column}'] = np.nan
                results_list.append(wsi_result)
                continue

            # 1. 检查是否有 Hull 轮廓数据
            hulls = all_wsi_hulls.get(wsi_id, [])
            if not hulls:
                print(f"警告: wsi_id '{wsi_id}' 没有有效的组织凸包 (Hulls) 数据，已跳过 PPM。")
                # ... (填充 NaN 逻辑)
                results_list.append(wsi_result)
                continue


            try:
                # 2. 定义 PPM 窗口 (基于所有点的边界框)
                x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
                y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

                # 传入 R
                r.assign("x_min_r", x_min)
                r.assign("x_max_r", x_max)
                r.assign("y_min_r", y_min)
                r.assign("y_max_r", y_max)

                r.assign("hulls_list_r", hulls)

                r.assign("DBSCAN_EPS", DBSCAN_EPS)

                # R 脚本: 创建组织 Masking 窗口 (owin)
                r_script_window = """
                            # 将每个 hull 顶点数组（N x 2 矩阵）转换为 list(x, y) 形式，
                            # 并将它们收集在一个列表中 (即 list of list(x,y))
                            hull_polygons <- lapply(hulls_list_r, function(h) {
                                return(list(x=c(h[,1], h[1,1]), y=c(h[,2], h[1,2])))
                            })

                            # PPM 窗口 (使用组织 Mask)
                            # 注意: polygonal 窗口要求多边形是闭合的，所以我们在上面加上了 h[1,1] 和 h[1,2] 来闭合多边形。
                            # 使用边界框和多边形列表创建窗口
                            win_ppm <- owin(xrange=c(x_min_r, x_max_r), 
                                            yrange=c(y_min_r, y_max_r),
                                            poly=hull_polygons) 

                            # 确保窗口是一个有效的 owin 对象
                            if (is.null(win_ppm)) {
                                # 如果多边形创建失败，尝试回退到简单的矩形窗口
                                win_ppm <- owin(xrange=c(x_min_r, x_max_r), 
                                                yrange=c(y_min_r, y_max_r))
                            }
                            """
                r(r_script_window)
                # ------------------------------------

                # 3. 创建 PPM 的点模式对象 (ppp) - 使用 win_ppm 作为窗口
                r.assign("target_x_r", target_points[:, 0])
                r.assign("target_y_r", target_points[:, 1])

                # R 脚本: 创建目标点模式
                r_script_ppp = """
                                # 使用组织 Mask 作为观察窗口
                                ppp_target <- ppp(x=target_x_r, y=target_y_r, window=win_ppm) 
                                """
                r(r_script_ppp)

                # 4. 创建协变量图像 (Z 表面) - 使用 win_ppm 作为窗口
                r.assign("all_x_r", all_points[:, 0])
                r.assign("all_y_r", all_points[:, 1])
                r.assign("cov_z_r", all_points[:, 2])

                r_script_cov_ppp = """
                                # 使用所有点和协变量值创建点模式对象，窗口为组织 Mask
                                ppp_cov <- ppp(x=all_x_r, y=all_y_r, window=win_ppm, marks=cov_z_r) 
                                # 使用 DBSCAN_EPS 作为 sigma，对齐聚类尺度
                                im_cov_surface <- Smooth(ppp_cov, sigma=DBSCAN_EPS) 
                                """
                r(r_script_cov_ppp)

                # 5. PPM 拟合
                r_script_ppm = f"""
                                # 拟合非齐次 PPM: log(lambda) = alpha + beta * Z
                                # 使用 ppp_target (已限制在组织 Mask内) 和 im_cov_surface (也限制在 Mask内)
                                ppm_fit <- ppm(ppp_target ~ im_cov_surface)

                                # 提取系数和对数似然
                                coefs <- summary(ppm_fit)$coefs
                                beta_z <- coefs["im_cov_surface", "Estimate"]
                                p_value_z <- coefs["im_cov_surface", "Pr(>|z|)"]
                                log_lik <- logLik(ppm_fit)[1]
                                """
                r(r_script_ppm)

                # 6. 提取结果
                beta_z_r = r['beta_z']
                p_value_z_r = r['p_value_z']
                log_lik_r = r['log_lik']

                if beta_z_r is R_NULL or p_value_z_r is R_NULL or log_lik_r is R_NULL:
                    beta_z = beta_z_r[0]
                    p_value_z = None
                    log_lik = log_lik_r[0]
                else:
                    # 安全提取值
                    beta_z = beta_z_r[0]
                    p_value_z = p_value_z_r[0]
                    log_lik = log_lik_r[0]

                # 7. 可视化 (可选)
                if if_plot:
                    # 这部分代码在实际生产环境中可能需要注释或去除，以避免图形输出带来的开销
                    r_plot = f"""
                    plot(im_cov_surface, main='Covariate Z Surface for WSI {wsi_id}')
                    plot(ppp_target, add=TRUE)
                    """
                    r(r_plot)

                # 8. 填充结果字典
                wsi_result[f'PPM_Beta_Z_{object_type}_{covariate_column}'] = beta_z
                wsi_result[f'PPM_P_Value_Z_{object_type}_{covariate_column}'] = p_value_z
                wsi_result[f'PPM_LogLikelihood_{object_type}_{covariate_column}'] = log_lik

                results_list.append(wsi_result)

            except Exception as e:
                print(f"处理 wsi_id '{wsi_id}' PPM 拟合时发生错误: {e}")
                # 发生任何错误时，同样填充NaN
                for feature in ['Beta_Z', 'P_Value_Z', 'LogLikelihood']:
                    wsi_result[f'PPM_{feature}_{object_type}_{covariate_column}'] = np.nan
                results_list.append(wsi_result)

    if not results_list:
        return pd.DataFrame()

    return pd.DataFrame(results_list)



if __name__ == '__main__':

    input_data_path = "/home/yangy50/project/pywasphe/PyWasPhe/result/concate_csv_topology/"
    result_path="/home/yangy50/project/pywasphe/PyWasPhe/result/topology_feature_1st_order.csv"


    filename_map = {

        'arteries': 'arteriesarterioles_Features.csv',

        'globally_sclerotic_glomeruli': 'globally_sclerotic_glomeruli_Features.csv',

        'non_globally_sclerotic_glomeruli': 'non_globally_sclerotic_glomeruli_Features.csv',

        'tubules': 'tubules_Features.csv'

    }


    #
    # objects_to_process = ['tubules']
    objects_to_process = ['arteries', 'tubules', 'globally_sclerotic_glomeruli', 'non_globally_sclerotic_glomeruli']


    all_patients_data = []

    for patient_folder in tqdm(os.listdir(input_data_path), desc="处理病患文件夹"):

        patient_folder_path = os.path.join(input_data_path, patient_folder)

        # 遍历我们需要分析的每一种对象类型
        all_data = []

        for obj_type in objects_to_process:

            csv_filename = filename_map[obj_type]

            csv_path = os.path.join(patient_folder_path, csv_filename)


            # 检查CSV文件是否存在

            if os.path.exists(csv_path):

                try:

                    # 读取CSV文件

                    df = pd.read_csv(csv_path)

                    # 添加一列来标记这个数据属于哪种对象类型

                    df['object_type'] = obj_type

                    all_data.append(df)

                except Exception as e:

                    print(f"错误: 无法读取或处理文件 {csv_path}。错误信息: {e}")


        # 将所有读取到的数据合并成一个大的DataFrame

        combined_df = pd.concat(all_data, ignore_index=True)

        other_col_df=combined_df.copy()

        ppm_covariate_name = 'Sum Distance Transform By Object Area Nuclei'
        # 筛选出我们需要的列，避免内存占用过大

        required_cols = ['wsi_id', 'topology_x', 'topology_y', 'object_type',ppm_covariate_name]


        combined_df = combined_df[required_cols]



        # 按 'wsi_id' 和 'object_type' 进行分组

        grouped = combined_df.groupby(['wsi_id', 'object_type'])

        wsi_ids = combined_df['wsi_id'].unique()

        ann_feature_df = calculate_ann_features(grouped, objects_to_process,wsi_ids)

        # get the tissue area （replace this one）
        # area_df=get_tissue_area(combined_df)

        # 1. 【修改】计算组织面积，并同时获取带有 Cluster ID 的点数据
        area_df, clustered_combined_df,all_wsi_hulls = get_tissue_area_and_clusters(combined_df.copy())

        # 2. 【新增】计算质心分析特征 (Mean Center, SD, SDE)
        centrography_features_df = calculate_centrography_features(clustered_combined_df.copy(), ['tubules', 'arteries'])

        # 3. 检查 area_df 是否为空，以防某些WSI的 area 计算失败
        if area_df.empty:
            print(f"警告: 病人 {patient_folder} 未能计算出任何有效的组织面积，已跳过后续特征计算。")
            continue

        # 4. 【修改】将全局密度特征的计算基于 area_df
        global_density_feature = calculate_global_density(combined_df.copy(), area_df, ['tubules'])

        # 强度与协变量的建模 (Intensity Modeling with Covariates)
        # 协变量设置

        ppm_object_type = ["tubules"]

        # 1. 【新增】计算 PPM 强度建模特征
        # 注意: 这里我们传入 combined_df，因为它包含 PPM 需要的协变量列
        ppm_feature_df = calculate_ppm_features_r_backend(
            combined_df.copy(),
            area_df,
            all_wsi_hulls,
            ppm_object_type,
            ppm_covariate_name,
            if_plot=True  # 默认关闭绘图
        )


        print(1)