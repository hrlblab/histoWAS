import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import math


# --- 1. 最终修正版 - 曼哈顿图 (已添加排序功能) ---
def plot_manhattan_feature_categorized(regressions, *, thresh, save='', save_format='png'):
    """
    绘制带分类的特征曼哈顿图，并按显著性从高到低排序。
    - Pathomics: 蓝色圆点
    - Spatial: 红色方块
    """
    # ----------------> 修改开始 <----------------

    # 复制DataFrame以避免修改原始数据
    regressions_sorted = regressions.copy()

    # <--- 新增代码: 按 "-log(p)" 列降序排序 ---
    regressions_sorted.sort_values(by='"-log(p)"', ascending=False, inplace=True)

    # <--- 新增代码: 重置索引，确保X轴位置从0开始连续排列 ---
    regressions_sorted.reset_index(drop=True, inplace=True)

    # ----------------> 修改结束 <----------------

    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {'Pathomics': 'blue', 'Spatial': 'red'}
    markers = {'Pathomics': 'o', 'Spatial': 's'}

    # 注意：现在使用排好序的 regressions_sorted 进行绘图
    for category, group_df in regressions_sorted.groupby('Category'):
        ax.scatter(
            group_df.index,
            group_df['"-log(p)"'],
            color=palette.get(category, 'gray'),
            marker=markers.get(category, 'x'),
            s=20,
            label=category
        )

    ax.axhline(y=-math.log10(thresh), color='red', linestyle='--', label='Threshold (-log10)')

    # 使用排好序的 regressions_sorted 来设置X轴标签
    xs = list(range(len(regressions_sorted)))
    ax.set_xticks(xs)
    ax.set_xticklabels(regressions_sorted["Feature"], rotation=90, fontsize=6)

    ax.set_xlabel("Feature (Sorted by Significance)")  # 更新X轴标签以反映排序
    ax.set_ylabel("-log10(p)")
    ax.set_title("Manhattan Plot (Features by Category)")
    ax.legend(title='Feature Category')
    plt.tight_layout()

    if save:
        plt.savefig(save, format=save_format, dpi=300)
        plt.close()
    else:
        plt.show()



# --- 2. 最终修正版 - 效应量图 ---
def plot_effect_size_feature_categorized(regressions, *, thresh, save='', save_format='png'):
    """
    绘制带分类的特征效应量图，保持原始图表样式。
    - Pathomics: 原始颜色(aqua)圆点
    - Spatial: 新颜色(vermillion)方块
    """
    sig_df = regressions[regressions["p-val"] < thresh].copy()
    if sig_df.empty:
        print("没有特征通过阈值筛选，无法绘制效应量图。")
        return

    # 解析置信区间的逻辑保持不变
    lower_bounds, upper_bounds = [], []
    for ci in sig_df["Conf-interval beta"]:
        try:
            low, high = map(float, ci.strip("[]").split(","))
            lower_bounds.append(low)
            upper_bounds.append(high)
        except:
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
    sig_df["lower"] = lower_bounds
    sig_df["upper"] = upper_bounds

    sig_df.sort_values(by="beta", inplace=True)
    sig_df.reset_index(drop=True, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 6)) # 保持原始尺寸
    y_coord = 1
    text_size = 6

    # 定义颜色和形状映射，Pathomics使用原始颜色
    palette = {'Pathomics': 'blue', 'Spatial': 'red'}
    markers = {'Pathomics': 'o', 'Spatial': 's'}

    for _, row in sig_df.iterrows():
        category = row["Category"]
        beta_val = row["beta"]
        color = palette.get(category, 'gray')
        marker = markers.get(category, 'x')

        # 绘制置信区间和数据点
        ax.plot([row["lower"], row["upper"]], [y_coord, y_coord], color=color)
        ax.plot(beta_val, y_coord, marker=marker, color=color, fillstyle='full', markeredgewidth=0)

        ha = 'left' if beta_val > 0 else 'right'
        ax.text(beta_val, y_coord, row["Feature"], ha=ha, va='center', fontsize=text_size)
        y_coord += 15

    # 手动创建图例
    legend_handles = [
        mlines.Line2D([], [], color=palette['Pathomics'], marker=markers['Pathomics'], linestyle='None', label='Pathomics'),
        mlines.Line2D([], [], color=palette['Spatial'], marker=markers['Spatial'], linestyle='None', label='Spatial')
    ]
    ax.legend(handles=legend_handles, title="Feature Category")

    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel("Beta (Effect Size)")
    ax.set_yticks([])
    ax.set_title("Effect Size Plot (Significant Features)")
    plt.tight_layout()

    if save:
        plt.savefig(save, format=save_format, dpi=300)
        plt.close()
    else:
        plt.show()

# --- 3. 最终修正版 - 火山图 ---
def plot_volcano_feature_categorized(regressions, *, p_thresh=0.05, save='', save_format='png'):
    """
    绘制火山图，用颜色表示显著性，用形状表示类别。
    """
    # 恢复原始的Bonferroni/FDR阈值计算逻辑
    def get_bon_thresh(p_values, alpha=0.05):
        return alpha / sum(np.isfinite(p_values))
    def get_fdr_thresh(p_values, alpha=0.05):
        sn = np.sort(p_values)
        sn = sn[np.isfinite(sn)]
        if not sn.size: return 1.0
        for i in range(len(sn) - 1, -1, -1):
            p_crit = alpha * float(i + 1) / float(len(sn))
            if sn[i] <= p_crit:
                return sn[i]
        return sn[0] if sn.size else 1.0

    pvals = regressions["p-val"].values
    bon_threshold = get_bon_thresh(pvals)
    fdr_threshold = get_fdr_thresh(pvals)

    # 准备颜色（显著性）和形状（类别）
    colors = []
    markers = []
    labels = []
    marker_map = {'Pathomics': 'o', 'Spatial': 's'}

    for i, row in regressions.iterrows():
        p = row["p-val"]
        # 恢复原始的颜色逻辑
        if p < bon_threshold:
            colors.append("gold")
            labels.append(row["Feature"])
        elif p < fdr_threshold:
            colors.append("midnightblue")
            labels.append(row["Feature"])
        else:
            colors.append("slategray")
            labels.append("")
        # 新增的形状逻辑
        markers.append(marker_map.get(row["Category"], 'x'))

    fig, ax = plt.subplots(figsize=(8, 6)) # 恢复原始尺寸

    # 逐点绘制以应用不同的形状
    for i in range(len(regressions)):
        ax.scatter(regressions["beta"].iloc[i], regressions['"-log(p)"'].iloc[i],
                   c=colors[i], marker=markers[i], s=20) # 恢复原始点大小

    # 恢复原始的标签逻辑
    for i, label in enumerate(labels):
        if label != "":
            ax.text(regressions["beta"].iloc[i], regressions['"-log(p)"'].iloc[i], label,
                    fontsize=6, rotation=45, va='bottom')

    ax.set_xlabel("Beta (Effect Size)")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Volcano Plot (Color=Significance, Shape=Category)")
    ax.axvline(x=0, color="black", linestyle="--")

    # 更新图例以同时解释颜色和形状
    color_legend = [
        mlines.Line2D([], [], color='gold', marker='o', linestyle='None', label='Bonferroni significant'),
        mlines.Line2D([], [], color='midnightblue', marker='o', linestyle='None', label='FDR significant'),
        mlines.Line2D([], [], color='slategray', marker='o', linestyle='None', label='Not significant')
    ]
    shape_legend = [
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None', label='Pathomics'),
        mlines.Line2D([], [], color='gray', marker='s', linestyle='None', label='Spatial')
    ]
    first_legend = ax.legend(handles=color_legend, loc='upper right', title="Significance")
    ax.add_artist(first_legend)
    ax.legend(handles=shape_legend, loc='upper left', title="Category")

    plt.tight_layout()
    if save:
        plt.savefig(save, format=save_format, dpi=300)
        plt.close()
    else:
        plt.show()
