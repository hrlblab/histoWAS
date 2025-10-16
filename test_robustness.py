import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# 1. RPY2 设置 (与您之前的代码相同)
# =============================================================================
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# 导入R包
try:
    base = importr('base')
    stats = importr('stats')
    spatstat_geom = importr('spatstat.geom')
    spatstat_explore = importr('spatstat.explore')
    print("rpy2 和 spatstat 包加载成功。")
except ImportError:
    print("错误: R 或 spatstat 包未能成功加载。")
    print("请确保您已在R中安装 'spatstat' (install.packages('spatstat'))")
    exit()

# 激活rpy2的自动转换
# pandas2ri.activate()
# numpy2ri.activate()

# 忽略Pandas的FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


# =============================================================================
# 2. 从您代码中复制的核心计算函数
#    (无需任何修改，直接复用)
# =============================================================================
def calculate_g_features_r_backend(area_df: pd.DataFrame,
                                   combined_df: pd.DataFrame,
                                   process_list: list,
                                   distance_list: list) -> pd.DataFrame:
    """
    [R后端版] 使用rpy2调用R的spatstat包计算g-function (Pair Correlation Function)特征。
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

                x_min, x_max = group_df['topology_x'].min(), group_df['topology_x'].max()
                y_min, y_max = group_df['topology_y'].min(), group_df['topology_y'].max()

                r.assign("x_min_r", x_min)
                r.assign("x_max_r", x_max)
                r.assign("y_min_r", y_min)
                r.assign("y_max_r", y_max)

                # *** 新增代码: 将 distance_list 传递给 R ***
                r.assign("distance_list", distance_list)

                points1_df = group_df[group_df['object_type'] == object_type_1]
                r.assign("points1_x_r", points1_df['topology_x'].values)
                r.assign("points1_y_r", points1_df['topology_y'].values)

                r_script = """
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)
                pcf_result <- pcf(ppp1, correction="iso", r=seq(0, max(distance_list), length.out=101))
                pcf_result
                """
                g_result_r = r(r_script)

                # *** 修改代码: 使用方括号访问结果 ***
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


def calculate_k_features_r_backend(area_df: pd.DataFrame,
                                   combined_df: pd.DataFrame,
                                   process_list: list,
                                   distance_list: list,
                                   if_L: bool = True) -> pd.DataFrame:
    """
    [R后端版] 使用rpy2调用R的spatstat包计算K函数和L函数特征。
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

                area_info = area_df.loc[area_df['wsi_id'] == wsi_id]
                win_dims = area_info['window_dims'].iloc[0]

                r.assign("x_min_r", win_dims[0])
                r.assign("x_max_r", win_dims[1])
                r.assign("y_min_r", win_dims[2])
                r.assign("y_max_r", win_dims[3])

                # *** 新增代码: 将 distance_list 传递给 R ***
                r.assign("distance_list", distance_list)

                points1_df = group_df[group_df['object_type'] == object_type_1]
                r.assign("points1_x_r", points1_df['topology_x'].values)
                r.assign("points1_y_r", points1_df['topology_y'].values)

                r_script = f"""
                win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
                ppp1 <- ppp(x=points1_x_r, y=points1_y_r, window=win)
                k_result <- Kest(ppp1, correction="iso", r=seq(0, max(distance_list), length.out=101))
                k_result
                """
                k_result_r = r(r_script)

                # *** 修改代码: 使用方括号访问结果 ***
                r_distances = k_result_r['r']
                r_k_values = k_result_r['iso']

                k_values_at_distances = np.interp(distance_list, r_distances, r_k_values, left=np.nan, right=np.nan)

                if if_L:
                    k_for_l_calc = np.maximum(0, k_values_at_distances)
                    l_values_at_distances = np.sqrt(k_for_l_calc / np.pi) - np.array(distance_list)

                for i, d in enumerate(distance_list):
                    if if_L:
                        l_feature_name = f"{object_type_1}_{object_type_2}_distance_{d}_L_function"
                        wsi_result[l_feature_name] = l_values_at_distances[i]
                results_list.append(wsi_result)

            except Exception as e:
                print(f"处理 wsi_id '{wsi_id}' 时发生未知错误: {e}")
                for d in distance_list:
                    wsi_result[f"{object_type_1}_{object_type_2}_distance_{d}_L_function"] = np.nan
                results_list.append(wsi_result)

    return pd.DataFrame(results_list) if results_list else pd.DataFrame()


# =============================================================================
# 3. 新增的模拟与验证功能函数
# =============================================================================
def simulate_csr_points(area_dims, density):
    """
    在指定区域内生成符合CSR(泊松点过程)的二维点云。
    """
    area = area_dims[0] * area_dims[1]
    # 根据泊松分布确定点的总数
    num_points = np.random.poisson(area * density)
    # 在区域内均匀生成点的坐标
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
    # 将子区域的点坐标平移到以(0,0)为起点，以便后续处理
    sub_points[:, 0] -= x0
    sub_points[:, 1] -= y0

    return sub_points, (0, sample_dims[0], 0, sample_dims[1])


def get_simulation_envelope(sample_points, window_dims, func_name='Lest', n_sim=99):
    """
    使用spatstat的envelope功能计算置信区间。
    """
    print(f"正在为 {func_name} 计算理论置信区间...")
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        r.assign("points_x_r", sample_points[:, 0])
        r.assign("points_y_r", sample_points[:, 1])
        r.assign("x_min_r", window_dims[0])
        r.assign("x_max_r", window_dims[1])
        r.assign("y_min_r", window_dims[2])
        r.assign("y_max_r", window_dims[3])
        r.assign("func_r", func_name)
        r.assign("n_sim_r", n_sim)

        r_script = f"""
        win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
        ppp_obj <- ppp(x=points_x_r, y=points_y_r, window=win)
        env <- envelope(ppp_obj, fun={func_name}, nsim=n_sim_r, rank=1, correction="iso")
        env
        """
        env_result = r(r_script)

        # *** 修改代码: 将 .rx2() 替换为方括号 ***
        r_vals = env_result['r']
        lo_vals = env_result['lo']
        hi_vals = env_result['hi']

    print("置信区间计算完成。")
    return r_vals, lo_vals, hi_vals


def plot_results(distance_list, results, envelope, theoretical_val, title, ylabel, filename):
    """
    绘制模拟结果的可视化图表。
    """
    r_env, lo_env, hi_env = envelope

    plt.figure(figsize=(10, 8))

    # 1. 绘制所有模拟曲线
    for i, res_series in enumerate(results):
        label = 'Simulation Runs' if i == 0 else ''
        plt.plot(distance_list, res_series, color='cornflowerblue', alpha=0.2, label=label)

    # 2. 绘制置信区间
    plt.fill_between(r_env, lo_env, hi_env, color='grey', alpha=0.5, label='95% Simulation Envelope')

    # 3. 绘制理论期望线
    plt.axhline(y=theoretical_val, color='red', linestyle='--', linewidth=2,
                label=f'Theoretical Value = {theoretical_val}')

    plt.title(title, fontsize=16)
    plt.xlabel('Distance (r)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)

    # 合并图例，避免重复
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(filename, dpi=300)
    print(f"结果图已保存至: {filename}")
    plt.show()


# =============================================================================
# 4. 主执行逻辑
# =============================================================================
# =============================================================================
# 4. 主执行逻辑 (修正版)
# =============================================================================
if __name__ == '__main__':
    # --- 参数配置 ---
    LARGE_AREA_DIMS = (10000, 10000)  # "基底"点云的尺寸 (width, height)
    SAMPLE_WINDOW_DIMS = (1000, 1000)  # 每次采样的窗口尺寸 (width, height)
    POINT_DENSITY = 0.0005  # 点的密度 (每平方单位面积的点数)
    N_SAMPLES = 100  # 重复采样的次数

    # 用于评估的距离列表 (更密集以获得平滑曲线)
    DISTANCE_LIST = np.linspace(0, 400, 101)

    # --- 步骤 1: 生成大型CSR“基底”点云 ---
    print("--- 步骤 1: 生成CSR点云 ---")
    csr_points = simulate_csr_points(LARGE_AREA_DIMS, POINT_DENSITY)

    # --- 步骤 2: 循环采样并计算 ---
    print(f"\n--- 步骤 2: 开始 {N_SAMPLES} 次重复采样与计算 ---")
    all_l_function_results = []
    all_g_function_results = []

    first_sample_points = None  # 用于计算置信区间
    first_sample_window = None

    for i in tqdm(range(N_SAMPLES), desc="模拟进度"):
        # 从大点云中采样一个子区域
        sample_points, window_dims = sample_subregion(csr_points, SAMPLE_WINDOW_DIMS, LARGE_AREA_DIMS)

        if i == 0:
            first_sample_points = sample_points
            first_sample_window = window_dims

        # 准备输入数据，使其符合您的函数要求
        wsi_id = f'sim_{i}'
        combined_df = pd.DataFrame({
            'wsi_id': wsi_id,
            'topology_x': sample_points[:, 0],
            'topology_y': sample_points[:, 1],
            'object_type': 'simulated_points'
        })

        area_df = pd.DataFrame({
            'wsi_id': [wsi_id],
            'tissue_area': [SAMPLE_WINDOW_DIMS[0] * SAMPLE_WINDOW_DIMS[1]],
            'window_dims': [window_dims]  # 传递精确的窗口信息
        })

        # 调用您的函数计算L-function
        l_result_df = calculate_k_features_r_backend(
            area_df, combined_df, ["simulated_points", "simulated_points"], DISTANCE_LIST, if_L=True
        )
        if not l_result_df.empty:
            l_values = l_result_df.drop('wsi_id', axis=1).iloc[0].values
            all_l_function_results.append(l_values)

        # 调用您的函数计算g-function
        g_result_df = calculate_g_features_r_backend(
            area_df, combined_df, ["simulated_points", "simulated_points"], DISTANCE_LIST
        )
        if not g_result_df.empty:
            g_values = g_result_df.drop('wsi_id', axis=1).iloc[0].values
            all_g_function_results.append(g_values)

    # --- 步骤 3: 计算理论置信区间 ---
    print("\n--- 步骤 3: 计算理论置信区间 ---")

    # *** 核心修改部分 ***
    # 1. 先获取 K-function 的置信区间
    k_envelope = get_simulation_envelope(first_sample_points, first_sample_window, func_name='Kest', n_sim=99)

    # 2. 手动将 K-function 的置信区间转换为 L-function 的置信区间
    r_env, lo_k, hi_k = k_envelope
    lo_l = np.sqrt(np.maximum(0, lo_k) / np.pi) - r_env
    hi_l = np.sqrt(np.maximum(0, hi_k) / np.pi) - r_env
    l_envelope = (r_env, lo_l, hi_l)

    # g-function 的计算方式保持不变
    g_envelope = get_simulation_envelope(first_sample_points, first_sample_window, func_name='pcf', n_sim=99)

    # --- 步骤 4: 可视化结果 ---
    print("\n--- 步骤 4: 生成结果图表 ---")

    # 绘制 L-function 结果
    plot_results(
        distance_list=DISTANCE_LIST,
        results=all_l_function_results,
        envelope=l_envelope,
        theoretical_val=0,
        title='Robustness Test for L-function on CSR Data',
        ylabel='L(r)',
        filename='robustness_L_function.png'
    )

    # 绘制 g-function 结果
    plot_results(
        distance_list=DISTANCE_LIST,
        results=all_g_function_results,
        envelope=g_envelope,
        theoretical_val=1,
        title='Robustness Test for g-function on CSR Data',
        ylabel='g(r)',
        filename='robustness_g_function.png'
    )

    print("\n模拟与验证完成！")