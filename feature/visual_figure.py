import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path
from PIL import Image
import openslide
import os
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from matplotlib.lines import Line2D
import geopandas as gpd
from shapely.geometry import Point, Polygon as ShapelyPolygon
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter


try:
    base = importr('base')
    stats = importr('stats')
    spatstat_geom = importr('spatstat.geom')
    spatstat_explore = importr('spatstat.explore')
    print("rpy2 和 spatstat 包加载成功。")
except ImportError as e:
    print(f"错误: R 或 spatstat 包未能成功加载: {e}")
    print("请确保您已在R环境中安装 'spatstat' (在R中运行: install.packages('spatstat'))")
    exit()



# 导入pysal/esda用于空间自相关分析
try:
    from esda.moran import Moran_Local
    from libpysal.weights import KNN
    import pointpats
except ImportError:
    print("错误：请安装 PySAL 相关的库: pip install geopandas libpysal esda pointpats")
    exit()

# --- 您原始代码中的设置和函数 ---

# 这一行在许多系统上是处理大图所必需的
Image.MAX_IMAGE_PIXELS = None

# 假设DBSCAN参数已经调好
DBSCAN_EPS = 5000
DBSCAN_MIN_SAMPLES = 15


def get_dbscan_clusters_and_hulls(points):
    """
    对输入的点进行DBSCAN聚类，并为每个簇计算凸包。
    返回一个字典，键是簇ID，值是包含点和凸包的字典。
    """
    print(f"正在对 {len(points)} 个点进行DBSCAN聚类 (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1).fit(points)

    unique_labels = sorted(set(db.labels_))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"聚类完成，发现 {n_clusters} 个簇。")
    if -1 in unique_labels:
        noise_points_count = np.sum(db.labels_ == -1)
        print(f"另外有 {noise_points_count} 个点被识别为噪声。")

    clusters = {}
    for label in unique_labels:
        if label == -1:
            continue  # 我们不处理噪声点

        cluster_mask = (db.labels_ == label)
        cluster_points = points[cluster_mask]

        if len(cluster_points) < 3:
            print(f"警告: Cluster {label} 的点少于3个，无法计算凸包。")
            continue

        try:
            hull = ConvexHull(cluster_points)
            clusters[label] = {
                'points': cluster_points,
                'hull': hull,
                'indices': np.where(cluster_mask)[0]
            }
        except Exception as e:
            print(f"警告: 无法为 Cluster {label} 计算凸包: {e}")

    return clusters


# --- 新增的四个可视化分析函数 ---
def plot_kernel_density(ax, cluster_data, hull_object, scale_factor, title):
    """
    1. 组织内特征的强度/密度分布图 (KDE) - R/spatstat后端，最终闭合多边形修正版
    """
    print("      使用 R/spatstat 后端计算带有边缘校正的 KDE...")

    points_original = cluster_data
    hull_vertices_original = hull_object.points[hull_object.vertices]

    # --- 核心修改在这里 ---
    # 1. 闭合多边形：将第一个顶点追加到末尾
    first_vertex = hull_vertices_original[0, :].reshape(1, -1)
    closed_hull_vertices = np.vstack([hull_vertices_original, first_vertex])

    try:
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            # 2. 将闭合后的顶点数据传递给R
            r.assign("points_x_r", points_original[:, 0])
            r.assign("points_y_r", points_original[:, 1])
            r.assign("hull_vertices_x_r", closed_hull_vertices[:, 0])
            r.assign("hull_vertices_y_r", closed_hull_vertices[:, 1])

            # R脚本
            r_script = """
            # 使用嵌套列表结构，并传入闭合的多边形顶点
            win <- owin(poly=list(list(x=hull_vertices_x_r, y=hull_vertices_y_r)))

            # 检查窗口是否有效，如果无效则回退到矩形边界框
            if (!is.owin(win) || area.owin(win) == 0) {
                win <- owin(xrange=range(hull_vertices_x_r), yrange=range(hull_vertices_y_r))
            }

            ppp_obj <- ppp(x=points_x_r, y=points_y_r, window=win)

            # 自动选择一个合理的平滑带宽 (sigma)
            sigma <- min(diff(ppp_obj$window$xrange), diff(ppp_obj$window$yrange))/10
            kde_result <- density(ppp_obj, sigma=sigma, edge=TRUE, correction="iso")

            as.matrix(kde_result)
            """

            kde_matrix = r(r_script)

            kde_result_r = r('kde_result')
            xrange = kde_result_r.rx2('xrange')
            yrange = kde_result_r.rx2('yrange')

            kde_matrix_py = np.array(kde_matrix).T

            extent_scaled = [
                xrange[0] / scale_factor, xrange[1] / scale_factor,
                yrange[0] / scale_factor, yrange[1] / scale_factor
            ]

        im = ax.imshow(kde_matrix_py, cmap='viridis',
                       extent=extent_scaled,
                       origin='lower', aspect='equal')

        hull_vertices_scaled = hull_object.points[hull_object.vertices] / scale_factor
        hull_path = Path(hull_vertices_scaled)
        patch = PathPatch(hull_path, facecolor='none', transform=ax.transData)
        ax.add_patch(patch)
        im.set_clip_path(patch)

        ax.set_title(title, fontsize=20, pad=20)
        ax.set_aspect('equal', adjustable='box')
        print("      边缘校正的 KDE 计算和绘图完成。")

    except Exception as e:
        error_text = f"Edge-corrected KDE failed:\n{e}"
        ax.text(0.5, 0.5, error_text, ha='center', va='center', color='red', transform=ax.transAxes, fontsize=8)
        print(f"错误: 边缘校正的 KDE 失败: {e}")



def plot_kernel_density_old(ax, cluster_data, hull_object, scale_factor, title):
    """
    1. 组织内特征的强度/密度分布图 (Kernel Density Estimation) - 使用Scipy
    """
    points_scaled = cluster_data / scale_factor
    hull_vertices_scaled = hull_object.points[hull_object.vertices] / scale_factor

    # 1. 使用 scipy.stats.gaussian_kde 计算KDE
    # 它需要的数据格式是 [n_dimensions, n_points]
    x_coords = points_scaled[:, 0]
    y_coords = points_scaled[:, 1]
    xy_stacked = np.vstack([x_coords, y_coords])

    # 创建KDE对象，bw_method可以用来调整平滑度
    try:
        kde = gaussian_kde(xy_stacked, bw_method=0.3)
    except np.linalg.LinAlgError:
        # 如果点非常集中，可能导致矩阵不可逆，使用更小的bw_method重试
        kde = gaussian_kde(xy_stacked, bw_method='silverman')

    # 2. 创建一个网格来覆盖整个凸包区域
    xmin, ymin = hull_vertices_scaled.min(axis=0)
    xmax, ymax = hull_vertices_scaled.max(axis=0)
    grid_x, grid_y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    # 3. 在网格的每个点上评估KDE的值
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
    z_values = np.reshape(kde(positions).T, grid_x.shape)

    # 4. 使用imshow绘制KDE结果（热力图）
    # np.rot90()是必要的，因为imshow和mgrid的坐标原点习惯不同
    im = ax.imshow(np.rot90(z_values), cmap='viridis',
                   extent=[xmin, xmax, ymin, ymax],
                   aspect='equal')

    # 5. 使用凸包作为遮罩来裁剪图像
    hull_path = Path(hull_vertices_scaled)
    patch = PathPatch(hull_path, facecolor='none', transform=ax.transData)
    ax.add_patch(patch)
    im.set_clip_path(patch)

    ax.set_title(title, fontsize=20, pad=20)
    ax.set_aspect('equal', adjustable='box')


def plot_spatial_autocorrelation(ax, cluster_data_df, hull_object, scale_factor, title, covariate_name):
    """
    2. 空间自相关 (Moran's I / LISA Map)
    """
    points_scaled = cluster_data_df[['x_scaled', 'y_scaled']].values
    hull_vertices_scaled = hull_object.points[hull_object.vertices] / scale_factor

    # 创建 GeoDataFrame
    gdf = gpd.GeoDataFrame(
        cluster_data_df,
        geometry=gpd.points_from_xy(cluster_data_df.x_scaled, cluster_data_df.y_scaled)
    )

    # 计算空间权重 (使用最近的8个邻居)
    weights = KNN.from_dataframe(gdf, k=8)
    weights.transform = 'r'  # 行标准化

    # 计算 Local Moran's I
    lisa = Moran_Local(gdf[covariate_name], weights)

    # 绘制LISA聚类图
    # HH = 1, LL = 2, LH = 3, HL = 4
    # 红色代表高-高聚集，蓝色代表低-低聚集
    quadrant_colors = {1: 'red', 2: 'blue', 3: 'lightblue', 4: 'pink'}
    p_significant = 0.05

    # 绘制所有点作为背景
    ax.scatter(points_scaled[:, 0], points_scaled[:, 1], c='lightgray', s=10, alpha=0.5)

    # 仅绘制显著的点
    significant_mask = (lisa.p_sim <= p_significant)
    gdf_sig = gdf[significant_mask]
    lisa_quads_sig = lisa.q[significant_mask]

    if not gdf_sig.empty:
        for q, color in quadrant_colors.items():
            mask = (lisa_quads_sig == q)
            ax.scatter(gdf_sig[mask].geometry.x, gdf_sig[mask].geometry.y, c=color, s=50, edgecolor='black',
                       label=f'Quadrant {q}')

    # 绘制凸包边界
    hull_poly = Polygon(hull_vertices_scaled, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(hull_poly)

    ax.set_title(title, fontsize=20, pad=20)
    ax.set_aspect('equal', adjustable='box')


def plot_l_function(ax, cluster_data, hull_object, title):
    """
    3. L-function图 (R-spatstat后端，最终修正版)
    """
    print("      使用 R/spatstat 后端计算 L-function...")

    hull_vertices = hull_object.points[hull_object.vertices]
    bounding_box = [
        hull_vertices[:, 0].min(), hull_vertices[:, 0].max(),
        hull_vertices[:, 1].min(), hull_vertices[:, 1].max()
    ]

    try:
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            # 将Python数据传递给R
            r.assign("points_x_r", cluster_data[:, 0])
            r.assign("points_y_r", cluster_data[:, 1])
            r.assign("x_min_r", bounding_box[0])
            r.assign("x_max_r", bounding_box[1])
            r.assign("y_min_r", bounding_box[2])
            r.assign("y_max_r", bounding_box[3])

            # 编写并执行R脚本
            r_script = f"""
            win <- owin(xrange=c(x_min_r, x_max_r), yrange=c(y_min_r, y_max_r))
            ppp_obj <- ppp(x=points_x_r, y=points_y_r, window=win)
            env <- envelope(ppp_obj, fun=Lest, nsim=99, rank=1, correction="iso")
            env
            """
            env_result = r(r_script)

            # --- 核心修改：使用 NumPy recarray 的属性访问方式 ---
            # 将 .rx2('r') 修改为 .r
            r_distances = env_result.r
            observed_L = env_result.obs
            lower_bound = env_result.lo
            upper_bound = env_result.hi
            theoretical_L = env_result.theo

        # 绘制结果
        ax.plot(r_distances, observed_L, color='black', label='Observed L-function')
        ax.plot(r_distances, theoretical_L, 'r--', label='Expected (CSR)')
        ax.fill_between(r_distances, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Envelope')

        ax.set_title(title, fontsize=16, pad=10)
        ax.set_xlabel("Distance (d)")
        ax.set_ylabel("L(d)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        print("      L-function 计算和绘图完成。")

    except Exception as e:
        # 将中文报错改为英文，避免字体警告
        error_text = f"L-function calculation failed:\n{e}"
        ax.text(0.5, 0.5, error_text, ha='center', va='center', color='red', transform=ax.transAxes)
        print(f"错误: L-function 计算失败: {e}")


def plot_interpolation(ax, cluster_data_df, hull_object, scale_factor, title, covariate_name):
    """
    4. 创建连续的生物特征表面 (Spatial Interpolation)
    """
    points_scaled = cluster_data_df[['x_scaled', 'y_scaled']].values
    values = cluster_data_df[covariate_name].values
    hull_vertices_scaled = hull_object.points[hull_object.vertices] / scale_factor

    # 创建一个网格来覆盖整个凸包区域
    xmin, ymin = hull_vertices_scaled.min(axis=0)
    xmax, ymax = hull_vertices_scaled.max(axis=0)
    grid_x, grid_y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    # 进行插值
    grid_z = griddata(points_scaled, values, (grid_x, grid_y), method='cubic')

    # 绘制插值结果
    im = ax.imshow(grid_z.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='inferno', aspect='equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 使用凸包作为遮罩
    hull_path = Path(hull_vertices_scaled)
    patch = PathPatch(hull_path, facecolor='none')
    ax.add_patch(patch)
    im.set_clip_path(patch)

    # 绘制原始点
    ax.scatter(points_scaled[:, 0], points_scaled[:, 1], c='white', s=5, alpha=0.5)

    ax.set_title(title, fontsize=20, pad=20)


# --- 主执行流程 ---
# --- 主执行流程 (修正版) ---
if __name__ == '__main__':

    # --- 1. 数据加载和准备 (这部分保持不变) ---
    root_path = "/home/yangy50/project/pywasphe/PyWasPhe/KPMP_visual/S-2006-005000_PAS_1of2/"
    output_dir = os.path.join(root_path, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)

    # ... (从 filename_map 到 combined_df.rename 的所有代码都保持不变) ...
    filename_map = {
        'arteries': 'arteriesarterioles_Features.csv',
        'globally_sclerotic_glomeruli': 'globally_sclerotic_glomeruli_Features.csv',
        'non_globally_sclerotic_glomeruli': 'non_globally_sclerotic_glomeruli_Features.csv',
        'tubules': 'tubules_Features.csv'
    }
    wsi_file_name = "S-2006-005000_PAS_1of2.svs"
    ppm_covariate_name = 'Sum Distance Transform By Object Area Nuclei'
    scale_factor = 4

    wsi_full_path = os.path.join(root_path, wsi_file_name)
    try:
        wsi_slide = openslide.OpenSlide(wsi_full_path)
        image_width, image_height = wsi_slide.dimensions
        final_width = image_width // scale_factor
        final_height = image_height // scale_factor
        background_image = wsi_slide.get_thumbnail((final_width, final_height))
        wsi_slide.close()
    except Exception as e:
        print(f"错误：无法打开或处理WSI文件 '{wsi_full_path}': {e}")
        exit()

    all_data = []
    for obj_type, csv_filename in filename_map.items():
        csv_path = os.path.join(root_path, csv_filename)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df['object_type'] = obj_type
                all_data.append(df)
            except Exception as e:
                print(f"错误: 无法读取或处理文件 {csv_path}。错误信息: {e}")

    if not all_data:
        print("错误：没有成功加载任何CSV数据。")
        exit()

    combined_df = pd.concat(all_data, ignore_index=True)
    if 'topology_x' in combined_df.columns and 'topology_y' in combined_df.columns:
        combined_df.rename(columns={'topology_x': 'x', 'topology_y': 'y'}, inplace=True)

    # --- 2. DBSCAN聚类和凸包计算 (这部分保持不变) ---
    points_all = combined_df[['x', 'y']].values
    clusters = get_dbscan_clusters_and_hulls(points_all)

    # --- 3. 为每个簇生成四种可视化图 (这里的逻辑已修正) ---
    for label, cluster_info in clusters.items():
        print(f"\n--- 正在为 Cluster {label} 生成可视化图 ---")

        cluster_indices = cluster_info['indices']
        cluster_full_df = combined_df.iloc[cluster_indices].copy()
        tubules_df = cluster_full_df[cluster_full_df['object_type'] == 'tubules'].copy()
        if tubules_df.empty:
            print(f"Cluster {label} 中没有 'tubules' 对象，跳过分析。")
            continue

        tubules_df['x_scaled'] = tubules_df['x'] / scale_factor
        tubules_df['y_scaled'] = tubules_df['y'] / scale_factor
        tubules_points_original = tubules_df[['x', 'y']].values

        try:
            tubules_hull = ConvexHull(tubules_points_original)
        except Exception as e:
            print(f"无法为Cluster {label}中的tubules计算凸包，跳过此簇: {e}")
            continue

        # --- 创建一个2x2的子图布局 ---
        fig, axes = plt.subplots(2, 2, figsize=(24, 24))
        fig.suptitle(f"WSI Spatial Analysis for Cluster {label} (Object: Tubules)", fontsize=30)

        # --- 核心修改在这里 ---
        # 1. 定义哪些子图需要WSI背景
        wsi_axes = [axes[0, 0], axes[0, 1], axes[1, 1]]

        # 2. 只在需要背景的子图上绘制背景
        for ax in wsi_axes:
            ax.imshow(background_image, alpha=0.5, extent=(0, final_width, final_height, 0))
            hull_v_scaled = tubules_hull.points[tubules_hull.vertices] / scale_factor
            xmin, ymin = hull_v_scaled.min(axis=0)
            xmax, ymax = hull_v_scaled.max(axis=0)
            padding = 50
            ax.set_xlim(xmin - padding, xmax + padding)
            ax.set_ylim(ymax + padding, ymin - padding)
            ax.set_xticks([])
            ax.set_yticks([])

        # --- 逐一调用分析函数 ---
        print("  1. 生成核密度图...")
        plot_kernel_density(axes[0, 0], tubules_points_original, tubules_hull, scale_factor,
                            "1. Kernel Density Estimation (KDE)")

        print("  2. 生成空间自相关图 (LISA)...")
        plot_spatial_autocorrelation(axes[0, 1], tubules_df, tubules_hull, scale_factor,
                                     "2. Spatial Autocorrelation (LISA)", ppm_covariate_name)

        # 3. 在干净的子图 axes[1, 0] 上调用L-function绘图
        print("  3. 生成L-function图...")
        plot_l_function(axes[1, 0], tubules_points_original, tubules_hull, "3. L-Function Analysis")

        print("  4. 生成插值表面图...")
        plot_interpolation(axes[1, 1], tubules_df, tubules_hull, scale_factor, "4. Interpolated Feature Surface",
                           ppm_covariate_name)

        # 保存图像
        output_filename = os.path.join(output_dir, f'Cluster_{label}_Tubules_Analysis.png')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Cluster {label} 的分析图已保存至: {output_filename}")

    print("\n所有分析完成！")