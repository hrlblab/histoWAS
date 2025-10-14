import pandas as pd
from pycirclize import Circos
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import math

# =============================================================================
# 1. 数据准备
# =============================================================================
try:
    df = pd.read_csv('/media/yangy50/Elements/KPMP_new/pathomics.csv')
    if "-log(p)" in df.columns:
        df.rename(columns={'-log(p)': 'log_p_val'}, inplace=True)
except FileNotFoundError:
    print("CSV file not found. Using randomly generated data for demonstration.")
    data = {
        'Feature': [f'Feature_{i:02d}' for i in range(1, 81)], 'log_p_val': np.random.rand(80) * 5 + 1,
        'beta': np.random.randn(80) * 0.5, 'Category': ['Pathomics'] * 40 + ['Spatial'] * 40,
        'Sub_Category': (['Sub_P1'] * 10 + ['Sub_P2'] * 10 + ['Sub_P3'] * 10 + ['Sub_P4'] * 10 +
                         ['Sub_S1'] * 10 + ['Sub_S2'] * 10 + ['Sub_S3'] * 10 + ['Sub_S4'] * 10)
    }
    df = pd.DataFrame(data)

df['beta'] = df['beta'].fillna(0)
df = df.sort_values(by=['Category', 'Sub_Category']).reset_index(drop=True)

# =============================================================================
# 2. 颜色映射 (全局)
# =============================================================================
unique_sub_categories = df['Sub_Category'].unique()
colors = plt.get_cmap("Set1").colors + plt.get_cmap("Set2").colors
subcategory_colors = colors[:len(unique_sub_categories)]
subcat_color_map = {subcat: color for subcat, color in zip(unique_sub_categories, subcategory_colors)}


# =============================================================================
# 3. 定义独立的绘图函数
# =============================================================================
def create_circos_plot(df_subset, category_name):
    """为单个Category的数据创建一张Circos图"""
    print(f"--- Creating plot for '{category_name}' ---")

    # ==================== 最终优化参数 ====================
    # 旋钮 1: 使用一个超大画布，为文字提供极限的周长空间
    fig = plt.figure(figsize=(50, 50))
    # =======================================================

    sectors = df_subset['Sub_Category'].value_counts().to_dict()
    circos = Circos(sectors, space=5)

    R_OUTER_DATA = 80

    for sector in circos.sectors:
        color = subcat_color_map.get(sector.name, 'lightgrey')
        sector.rect(r_lim=(0, R_OUTER_DATA), facecolor=color, edgecolor="black", linewidth=0.5)

    # (颜色映射部分代码不变)
    global_beta_max = abs(df['beta']).max()
    beta_norm = plt.Normalize(vmin=-global_beta_max, vmax=global_beta_max)
    beta_cmap = plt.cm.bwr

    global_pval_min = df['log_p_val'].min()
    global_pval_max = df['log_p_val'].max()
    pval_norm = plt.Normalize(vmin=global_pval_min, vmax=global_pval_max)
    pval_cmap = plt.cm.viridis

    for sector in circos.sectors:
        sub_category_name = sector.name
        sector_df = df_subset[df_subset['Sub_Category'] == sub_category_name]

        features = sector_df['Feature'].tolist()
        beta_values = sector_df['beta'].tolist()
        pval_values = sector_df['log_p_val'].tolist()
        positions = [i + 0.5 for i in range(len(features))]

        labels_track = sector.add_track((R_OUTER_DATA + 3, 100))  # 让轨道离数据区近一点
        labels_track.axis(fc="none", ec="none")
        for i in range(len(features)):
            position = positions[i]
            label = features[i]
            radian = labels_track.x_to_rad(position)
            degree = math.degrees(radian)
            if 90 < degree < 270:
                rotation = degree + 90;
                ha = "right"
            else:
                rotation = degree - 90;
                ha = "left"

            # ==================== 最终优化参数 ====================
            # 旋钮 2: 使用一个极小的字号来避免重叠
            labels_track.text(label, x=position, r=R_OUTER_DATA + 5, size=3.5, rotation=rotation, ha=ha, va="center")
            # =======================================================

        # (热力图部分代码不变)
        heatmap_track = sector.add_track((65, R_OUTER_DATA), r_pad_ratio=0.01)
        r_bottom, r_top = heatmap_track.r_lim
        for i in range(len(features)):
            color = beta_cmap(beta_norm(beta_values[i]))
            heatmap_track.bar(x=[positions[i]], height=[r_top - r_bottom], bottom=r_bottom, width=1.0, color=color,
                              edgecolor=color)

        pval_heatmap_track = sector.add_track((50, 65), r_pad_ratio=0.01)
        r_bottom_p, r_top_p = pval_heatmap_track.r_lim
        for i in range(len(features)):
            color = pval_cmap(pval_norm(pval_values[i]))
            pval_heatmap_track.bar(x=[positions[i]], height=[r_top_p - r_bottom_p], bottom=r_bottom_p, width=1.0,
                                   color=color, edgecolor=color)

    circos.plotfig()
    fig = circos.ax.get_figure()
    fig.suptitle(f"Circos Plot for {category_name}", fontsize=50, y=0.98)  # 增大标题字号

    # (图例和颜色条部分代码不变，但增大了字号以匹配大画布)
    subcat_legend_elements = [Patch(facecolor=subcat_color_map.get(subcat), edgecolor='k', label=subcat) for subcat in
                              sectors.keys()]
    legend1 = fig.legend(handles=subcat_legend_elements, loc="center left", bbox_to_anchor=(0.92, 0.75),
                         title="Sub-Category", fontsize=25)
    fig.add_artist(legend1)

    sm_beta = plt.cm.ScalarMappable(cmap=beta_cmap, norm=beta_norm)
    cbar_ax_beta = fig.add_axes([0.93, 0.45, 0.02, 0.2])
    cbar_beta = fig.colorbar(sm_beta, cax=cbar_ax_beta, orientation='vertical')
    cbar_beta.set_label("Beta Value", size=25)

    sm_pval = plt.cm.ScalarMappable(cmap=pval_cmap, norm=pval_norm)
    cbar_ax_pval = fig.add_axes([0.93, 0.15, 0.02, 0.2])
    cbar_pval = fig.colorbar(sm_pval, cax=cbar_ax_pval, orientation='vertical')
    cbar_pval.set_label("-log(p-value)", size=25)

    fig.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.05)
    fig.savefig(f"circos_plot_{category_name}_optimized.png", bbox_inches='tight', dpi=300)
    fig.savefig(f"circos_plot_{category_name}.svg", bbox_inches='tight', format='svg')
    print(f"Plot saved to circos_plot_{category_name}_optimized.png")
    plt.close(fig)

# =============================================================================
# 4. 主流程
# =============================================================================
all_categories = df['Category'].unique()
for cat in all_categories:
    subset = df[df['Category'] == cat].copy()
    if not subset.empty:
        create_circos_plot(subset, cat)

print("\nAll plots have been generated successfully.")