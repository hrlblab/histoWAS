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
    # ... (示例数据) ...

# *** 新增：清洗beta列的缺失值 ***
df['beta'].fillna(0, inplace=True)

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

    # ==================== 美学调整“旋钮” ====================
    # --- 旋钮 1: 增大画布尺寸 ---
    fig = plt.figure(figsize=(20, 20))  # 比如增加到 (20, 20)

    sectors = df_subset['Sub_Category'].value_counts().to_dict()

    # --- 旋钮 3: 增大扇区间隔 ---
    circos = Circos(sectors, space=0)  # 比如增加到 15
    # =======================================================

    for sector in circos.sectors:
        color = subcat_color_map.get(sector.name, 'lightgrey')
        sector.axis(facecolor=color, edgecolor="black", linewidth=0.5)

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
                rotation = degree + 90 if style == "outward" else degree - 90
                ha = "right"
            else:
                rotation = degree - 90 if style == "outward" else degree + 90
                ha = "left"

            # --- 旋钮 2: 减小标签字号 ---
            labels_track.text(label, x=position, r=102, size=8, rotation=rotation, ha=ha, va="center")  # 比如减小到 3

        # ... (其他轨道代码不变) ...
        heatmap_track = sector.add_track((80, 88), r_pad_ratio=0.05)
        r_bottom, r_top = heatmap_track.r_lim
        norm = plt.Normalize(vmin=-abs(df['beta']).max(), vmax=abs(df['beta']).max())
        cmap = plt.cm.bwr
        for i, (_, row) in enumerate(sector_df.iterrows()):
            color = cmap(norm(row['beta']))
            heatmap_track.bar(x=[positions[i]], height=[r_top - r_bottom], bottom=r_bottom,
                              color=color, edgecolor="black", linewidth=0.5)

        bar_track = sector.add_track((60, 78))
        bar_track.bar(x=positions, height=sector_df['log_p_val'].tolist(), color="darkslateblue")

    circos.plotfig()
    fig = circos.ax.get_figure()
    fig.suptitle(f"Circos Plot for {category_name}", fontsize=20)

    # ... (图例和保存代码不变) ...
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