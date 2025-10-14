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
        'beta': np.random.randn(80) * 1.5, 'Category': ['Pathomics'] * 40 + ['Spatial'] * 40,
        'Sub_Category': (['Sub_P1'] * 10 + ['Sub_P2'] * 10 + ['Sub_P3'] * 10 + ['Sub_P4'] * 10 +
                         ['Sub_S1'] * 10 + ['Sub_S2'] * 10 + ['Sub_S3'] * 10 + ['Sub_S4'] * 10)
    }
    df = pd.DataFrame(data)

# <--- 修改 (解决 FutureWarning): 采用推荐的写法 ---
df['beta'] = df['beta'].fillna(0)

df = df.sort_values(by=['Category', 'Sub_Category']).reset_index(drop=True)

# =============================================================================
# 2. 颜色映射 (全局)
# =============================================================================
unique_sub_categories = df['Sub_Category'].unique()
subcategory_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sub_categories)))
subcat_color_map = {subcat: color for subcat, color in zip(unique_sub_categories, subcategory_colors)}


# =============================================================================
# 3. 定义独立的绘图函数
# =============================================================================
def create_circos_plot(df_subset, category_name):
    """为单个Category的数据创建一张Circos图"""
    print(f"--- Creating plot for '{category_name}' ---")

    # 美学调整
    fig = plt.figure(figsize=(20, 20))
    sectors = df_subset['Sub_Category'].value_counts().to_dict()
    circos = Circos(sectors, space=0)

    for sector in circos.sectors:
        color = subcat_color_map.get(sector.name, 'lightgrey')
        sector.axis(facecolor=color, edgecolor="black", linewidth=0.5)

    # 预先计算全局的beta值范围，确保所有图的颜色条一致
    global_beta_max = abs(df['beta']).max()
    norm = plt.Normalize(vmin=-global_beta_max, vmax=global_beta_max)
    cmap = plt.cm.bwr

    for sector in circos.sectors:
        sub_category_name = sector.name
        sector_df = df_subset[df_subset['Sub_Category'] == sub_category_name].copy()

        features = sector_df['Feature'].tolist()
        positions = [i + 0.5 for i in range(len(features))]
        sector_df['position'] = positions

        # 轨道 1: 特征标签
        labels_track = sector.add_track((101, 115))
        labels_track.axis(fc="none", ec="none")
        for _, row in sector_df.iterrows():
            position, label = row['position'], row['Feature']
            radian = labels_track.x_to_rad(position)
            degree = math.degrees(radian)

            style = "outward"
            if 90 < degree < 270:
                rotation = degree + 90
                ha = "right"
            else:
                rotation = degree - 90
                ha = "left"

            labels_track.text(label, x=position, r=102, size=8, rotation=rotation, ha=ha, va="center")

        # 轨道 2: beta 值热力图
        heatmap_track = sector.add_track((80, 95), r_pad_ratio=0.05)
        r_bottom, r_top = heatmap_track.r_lim

        for i, (_, row) in enumerate(sector_df.iterrows()):
            color = cmap(norm(row['beta']))
            # <--- 修改 (解决 TypeError): 将 width=[1.0] 改为 width=1.0 ---
            heatmap_track.bar(x=[positions[i]], height=[r_top - r_bottom], bottom=r_bottom,
                              width=1.0, color=color, edgecolor=color)

        # 轨道 3: -log(p) 柱状图
        bar_track = sector.add_track((60, 78))
        bar_track.bar(x=positions, height=sector_df['log_p_val'].tolist(), color="darkslateblue")

    circos.plotfig()
    fig = circos.ax.get_figure()
    fig.suptitle(f"Circos Plot for {category_name}", fontsize=20)

    # 图例和保存
    subcat_legend_elements = [Patch(facecolor=subcat_color_map.get(subcat), edgecolor='k', label=subcat)
                              for subcat in sectors.keys()]
    legend1 = fig.legend(handles=subcat_legend_elements, loc="center left",
                         bbox_to_anchor=(0.92, 0.7), title="Sub-Category", fontsize=10)
    fig.add_artist(legend1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar_ax = fig.add_axes([0.93, 0.35, 0.02, 0.25])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label("Beta Value", size=12)

    fig.subplots_adjust(right=0.88, top=0.95)
    fig.savefig(f"circos_plot_{category_name}.png", bbox_inches='tight', dpi=300)
    print(f"Plot saved to circos_plot_{category_name}.png")
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