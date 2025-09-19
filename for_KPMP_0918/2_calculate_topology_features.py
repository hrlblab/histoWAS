from rpy2.robjects import numpy2ri

import os

import pandas as pd

import numpy as np

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



# 导入R的基础包和spatstat

base = importr('base')

stats = importr('stats')

spatstat_geom = importr('spatstat.geom')

spatstat_explore = importr('spatstat.explore')




# 忽略Pandas在进行concat操作时可能出现的FutureWarning

warnings.simplefilter(action='ignore', category=FutureWarning)

DBSCAN_EPS = 3000


# min_samples: 一个点被视为核心点所需的邻域内的最小样本数。

# 通常可以从一个较高的值开始尝试，以确保聚类的鲁棒性。

DBSCAN_MIN_SAMPLES = 1


# 4. 估算Mask尺寸时，在最大坐标基础上增加的边距（padding）

MASK_PADDING = 200  # 单位与您的坐标系相同


def calculate_g_features_r_backend(area_df: pd.DataFrame,
                                   combined_df: pd.DataFrame,
                                   process_list: list,
                                   distance_list: list) -> pd.DataFrame:
    """
    [R后端版] 使用rpy2调用R的spatstat包计算g-function (Pair Correlation Function)特征。
    功能完整，包含边缘校正。
    """
    if not isinstance(process_list, list) or len(process_list) != 2:
        raise ValueError("process_list必须是一个包含两个字符串的列表。")

    object_type_1 = process_list[0]
    object_type_2 = process_list[1]
    is_univariate = (object_type_1 == object_type_2)

    results_list = []

    for wsi_id, group_df in combined_df.groupby('wsi_id'):
        wsi_result = {'wsi_id': wsi_id}

        # 使用上下文管理器来激活转换规则
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            try:
                # 验证面积信息是否存在
                area_info = area_df.loc[area_df['wsi_id'] == wsi_id]
                if area_info.empty:
                    print(f"警告: 在 area_df 中未找到 wsi_id '{wsi_id}' 的面积信息，已跳过。")
                    continue

                # 1. 检查整个WSI的点数是否过少
                if len(group_df) < 2:
                    print(f"警告: wsi_id '{wsi_id}' 的总点数 ({len(group_df)}) 不足，已跳过。")
                    for d in distance_list:
                        feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_g_function"
                        wsi_result[feature_name] = np.nan
                    results_list.append(wsi_result)
                    continue

                # 2. 从当前WSI的所有点中计算实际的边界框 (Bounding Box)
                x_min, x_max = group_df['topology_x'].min(), group_df['topology_x'].max()
                y_min, y_max = group_df['topology_y'].min(), group_df['topology_y'].max()

                # 将边界信息传递给R
                r.assign("x_min_r", x_min)
                r.assign("x_max_r", x_max)
                r.assign("y_min_r", y_min)
                r.assign("y_max_r", y_max)

                # 3. 检查特定类型的点数
                points1_df = group_df[group_df['object_type'] == object_type_1]
                n1 = len(points1_df)

                if n1 < 2:
                    print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_1}' 的点数 ({n1}) 不足，已跳过。")
                    for d in distance_list:
                        feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_g_function"
                        wsi_result[feature_name] = np.nan
                    results_list.append(wsi_result)
                    continue

                # 将点坐标传递给R
                r.assign("points1_x_r", points1_df['topology_x'].values)
                r.assign("points1_y_r", points1_df['topology_y'].values)

                if is_univariate:
                    r_script = """
                    # 根据实际点坐标范围创建窗口
                    win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                    ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)
                    # <-- KEY CHANGE: 调用 pcf 而不是 Kest
                    pcf(ppp1, correction="iso") 
                    """
                    g_result_r = r(r_script)

                else:  # is_bivariate
                    points2_df = group_df[group_df['object_type'] == object_type_2]
                    n2 = len(points2_df)
                    if n2 < 2:
                        print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_2}' 的点数 ({n2}) 不足，已跳过。")
                        for d in distance_list:
                            feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_g_function"
                            wsi_result[feature_name] = np.nan
                        results_list.append(wsi_result)
                        continue

                    all_x = pd.concat([points1_df['topology_x'], points2_df['topology_x']], ignore_index=True).values
                    all_y = pd.concat([points1_df['topology_y'], points2_df['topology_y']], ignore_index=True).values
                    all_marks = np.concatenate([np.repeat(object_type_1, n1), np.repeat(object_type_2, n2)])

                    r.assign("all_x_r", all_x)
                    r.assign("all_y_r", all_y)
                    r.assign("all_marks_r", all_marks)
                    r.assign("type1_r", object_type_1)
                    r.assign("type2_r", object_type_2)

                    r_script = """
                    # 根据实际点坐标范围创建窗口
                    win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                    marks_factor <- factor(all_marks_r)
                    ppp_marked <- ppp(x=all_x_r, y=all_y_r, window=win, marks=marks_factor)
                    # <-- KEY CHANGE: 调用 pcfcross 而不是 Kcross
                    pcfcross(ppp_marked, i=type1_r, j=type2_r, correction="iso")
                    """
                    g_result_r = r(r_script)

                # 从R的结果对象中提取数据 (spatstat的pcf和Kest结果结构类似)
                r_distances = g_result_r['r']
                r_g_values = g_result_r['iso']

                g_values_at_distances = np.interp(distance_list, r_distances, r_g_values, left=np.nan, right=np.nan)

                # 循环填充结果字典
                for i, d in enumerate(distance_list):
                    g_feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_g_function"
                    wsi_result[g_feature_name] = g_values_at_distances[i]

                results_list.append(wsi_result)

            except Exception as e:
                print(f"处理 wsi_id '{wsi_id}' 时发生未知错误: {e}")
                # 发生任何未知错误时，同样填充NaN
                for d in distance_list:
                    feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_g_function"
                    wsi_result[feature_name] = np.nan
                results_list.append(wsi_result)

    if not results_list:
        return pd.DataFrame()

    return pd.DataFrame(results_list)





def calculate_k_features_r_backend(area_df: pd.DataFrame,

                                   combined_df: pd.DataFrame,

                                   process_list: list,

                                   distance_list: list,

                                   if_L: bool = True) -> pd.DataFrame:

    """

    [R后端版] 使用rpy2调用R的spatstat包计算K函数特征。

    功能完整，包含边缘校正。(已修正numpy.ndarray转换问题)

    """

    if not isinstance(process_list, list) or len(process_list) != 2:

        raise ValueError("process_list必须是一个包含两个字符串的列表。")


    object_type_1 = process_list[0]

    object_type_2 = process_list[1]

    is_univariate = (object_type_1 == object_type_2)


    results_list = []


    for wsi_id, group_df in combined_df.groupby('wsi_id'):

        wsi_result = {'wsi_id': wsi_id}


        # 使用上下文管理器来激活转换规则

        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):

            try:

                # 验证面积信息是否存在，但不再用它计算窗口

                area_info = area_df.loc[area_df['wsi_id'] == wsi_id]

                if area_info.empty:

                    print(f"警告: 在 area_df 中未找到 wsi_id '{wsi_id}' 的面积信息，已跳过。")

                    continue


                # --- 从这里开始是关键修改 ---


                # 1. 检查整个WSI的点数是否过少

                if len(group_df) < 2:

                    print(f"警告: wsi_id '{wsi_id}' 的总点数 ({len(group_df)}) 不足，已跳过。")

                    for d in distance_list:

                        feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_K_function"

                        wsi_result[feature_name] = np.nan

                    results_list.append(wsi_result)

                    continue


                # 2. 从当前WSI的所有点中计算实际的边界框 (Bounding Box)

                x_min, x_max = group_df['topology_x'].min(), group_df['topology_x'].max()

                y_min, y_max = group_df['topology_y'].min(), group_df['topology_y'].max()


                # 将边界信息传递给R

                r.assign("x_min_r", x_min)

                r.assign("x_max_r", x_max)

                r.assign("y_min_r", y_min)

                r.assign("y_max_r", y_max)


                # 3. 检查特定类型的点数

                points1_df = group_df[group_df['object_type'] == object_type_1]

                n1 = len(points1_df)


                if n1 < 2:

                    print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_1}' 的点数 ({n1}) 不足，已跳过。")

                    for d in distance_list:

                        feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_K_function"

                        wsi_result[feature_name] = np.nan

                    results_list.append(wsi_result)

                    continue


                # 将点坐标传递给R

                r.assign("points1_x_r", points1_df['topology_x'].values)

                r.assign("points1_y_r", points1_df['topology_y'].values)


                if is_univariate:

                    r_script = """

                    # 根据实际点坐标范围创建窗口

                    win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))

                    ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)

                    Kest(ppp1, correction="iso") 

                    """

                    k_result_r = r(r_script)


                else:  # is_bivariate

                    points2_df = group_df[group_df['object_type'] == object_type_2]

                    n2 = len(points2_df)

                    if n2 < 2:

                        print(f"警告: wsi_id '{wsi_id}' 中 '{object_type_2}' 的点数 ({n2}) 不足，已跳过。")

                        for d in distance_list:

                            feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_K_function"

                            wsi_result[feature_name] = np.nan

                        results_list.append(wsi_result)

                        continue


                    all_x = pd.concat([points1_df['topology_x'], points2_df['topology_x']], ignore_index=True).values

                    all_y = pd.concat([points1_df['topology_y'], points2_df['topology_y']], ignore_index=True).values

                    all_marks = np.concatenate([np.repeat(object_type_1, n1), np.repeat(object_type_2, n2)])


                    r.assign("all_x_r", all_x)

                    r.assign("all_y_r", all_y)

                    r.assign("all_marks_r", all_marks)

                    r.assign("type1_r", object_type_1)

                    r.assign("type2_r", object_type_2)


                    r_script = """

                    # 根据实际点坐标范围创建窗口

                    win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))

                    marks_factor <- factor(all_marks_r)

                    ppp_marked <- ppp(x=all_x_r, y=all_y_r, window=win, marks=marks_factor)

                    Kcross(ppp_marked, i=type1_r, j=type2_r, correction="iso")

                    """

                    k_result_r = r(r_script)


                # 从R的结果对象中提取数据 (这部分逻辑不变)

                r_distances = k_result_r['r']

                r_k_values = k_result_r['iso']


                k_values_at_distances = np.interp(distance_list, r_distances, r_k_values)


                if if_L:

                    # 使用公式 L(d) = sqrt(K(d)/pi) - d

                    # 为避免负数开方（理论上K(d)非负），使用np.maximum确保输入不为负

                    k_for_l_calc = np.maximum(0, k_values_at_distances)

                    l_values_at_distances = np.sqrt(k_for_l_calc / np.pi) - np.array(distance_list)


                    # 循环填充结果字典

                for i, d in enumerate(distance_list):

                    # 填充 K 函数特征

                    k_feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_K_function"

                    wsi_result[k_feature_name] = k_values_at_distances[i]


                    # *** 新增：填充 L 函数特征 ***

                    if if_L:

                        l_feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_L_function"

                        wsi_result[l_feature_name] = l_values_at_distances[i]


                results_list.append(wsi_result)


            except Exception as e:

                print(f"处理 wsi_id '{wsi_id}' 时发生未知错误: {e}")

                # 发生任何未知错误时，同样填充NaN

                for d in distance_list:

                    feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_K_function"

                    wsi_result[feature_name] = np.nan

                results_list.append(wsi_result)


    if not results_list:

        return pd.DataFrame()


    return pd.DataFrame(results_list)





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



if __name__ == '__main__':

    input_data_path = "/media/yangy50/Elements/KPMP_new/concate_csv_topology/"
    result_path="/media/yangy50/Elements/KPMP_new/topology_feature.csv"


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


        # # 确保这是一个文件夹而不是文件

        # if not os.path.isdir(patient_folder_path):

        #     continue


        # 遍历我们需要分析的每一种对象类型
        all_data = []

        for obj_type in objects_to_process:

            # 检查我们是否知道这个对象类型对应的文件名

            # if obj_type not in filename_map:

            #     print(f"警告: 未知的对象类型 '{obj_type}'，已跳过。")

            #     continue


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


        # 筛选出我们需要的列，避免内存占用过大

        required_cols = ['wsi_id', 'topology_x', 'topology_y', 'object_type']


        combined_df = combined_df[required_cols]



        # 按 'wsi_id' 和 'object_type' 进行分组

        grouped = combined_df.groupby(['wsi_id', 'object_type'])

        wsi_ids = combined_df['wsi_id'].unique()

        ann_feature_df = calculate_ann_features(grouped, objects_to_process,wsi_ids)


        # get the tissue area

        area_df=get_tissue_area(combined_df)


        # get the K function

        # single

        tubules_K_feature=calculate_k_features_r_backend(area_df,combined_df.copy(),["tubules","tubules"],[100,200,300,400])


        # # multi object
        #
        # tubules_globally_glomeruli_K_feature=calculate_k_features_r_backend(area_df,combined_df.copy(),["tubules","globally_sclerotic_glomeruli"],[100,200,300,400])

        # 单变量 g-function
        tubules_g_feature = calculate_g_features_r_backend(area_df, combined_df.copy(), ["tubules", "tubules"],
                                                           [100, 200, 300, 400])
        #
        # # 双变量 g-function
        # tubules_globally_glomeruli_g_feature = calculate_g_features_r_backend(area_df, combined_df.copy(), ["tubules",
        #                                                                                                     "globally_sclerotic_glomeruli"],
        #                                                                       [100, 200, 300, 400])

        features_to_merge = [ann_feature_df, tubules_K_feature, tubules_g_feature]
        # 过滤掉空的DataFrame
        features_to_merge = [df for df in features_to_merge if not df.empty]

        if not features_to_merge:
            print(f"警告: 病人 {patient_folder} 没有计算出任何有效的特征，已跳过。")
            continue

            # 使用 reduce 和 pd.merge 将列表中的所有DataFrame合并
            # 使用 'outer' 合并方式，确保即使某个特征计算失败，WSI的信息也能被保留
        patient_merged_df = reduce(lambda left, right: pd.merge(left, right, on='wsi_id', how='outer'),
                                   features_to_merge)

        # 3. 添加 'Biopsy ID' 列
        patient_merged_df['Biopsy ID: '] = patient_folder

        # 4. 将处理好的当前病人的DataFrame添加到总列表中
        all_patients_data.append(patient_merged_df)
        print(f"--- 病人 {patient_folder} 处理完成 ---")


    if all_patients_data:
        final_topology_df = pd.concat(all_patients_data, ignore_index=True)

        final_topology_df.drop(columns=['wsi_id'], axis=1,inplace=True)
        ave_result=final_topology_df.groupby('Biopsy ID: ').mean()



        # 6. 定义最终输出路径并保存为CSV
        final_topology_df.to_csv(result_path, index=False)

        print("\n=======================================================")
        print(f"所有计算完成！最终结果已保存至: {result_path}")
        print("最终DataFrame的前几行:")
        print(final_topology_df.head())
        print("=======================================================")
    else:
        print("\n--- 未处理任何数据，无法生成最终的 topology.csv 文件。---")
