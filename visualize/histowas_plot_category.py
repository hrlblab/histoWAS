import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import math
import pandas as pd


# --- 1. 最终修正版 - 曼哈顿图 (已添加排序功能) ---
def plot_manhattan_feature_categorized(regressions, *, thresh, show_num=None, save='', save_format='png'):
    """
    绘制带分类的特征曼哈顿图，并按显著性从高到低排序。
    - Pathomics: 蓝色圆点
    - Spatial: 红色方块
    """
    # ----------------> 修改开始 <----------------

    # 复制DataFrame以避免修改原始数据
    regressions_sorted = regressions.copy()

    if show_num is not None:
        # 1. 找出所有显著的特征 (p-val < thresh)
        sig_df = regressions_sorted[regressions_sorted["p-val"] < thresh]

        # 2. 找出所有非显著的特征 (p-val >= thresh)
        non_sig_df = regressions_sorted[regressions_sorted["p-val"] >= thresh]

        # 3. 对非显著特征按 "-log(p)" 降序排序，并取出前 show_num 个
        top_non_sig_df = non_sig_df.nlargest(show_num, columns='"-log(p)"')

        # 4. 合并显著特征和Top-N非显著特征
        regressions_sorted = pd.concat([sig_df, top_non_sig_df])

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


# --- 2. 最终修正版 (已修改) ---
def plot_effect_size_feature_categorized(regressions, *, thresh, save='', save_format='png'):
    """
    绘制带分类的特征效应量图（修改版）。
    - 解决了文字重叠问题。
    - 增加了特征间的垂直间距。
    - 将特征名称移到了置信区间线的正上方。
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

    # --- 修改点 开始 ---

    # 1. 动态调整图表高度，以容纳所有特征
    # 假设每个特征需要0.4英寸的高度，再加上2英寸的边距
    num_features = len(sig_df)
    # 确保图表有最小高度，同时也随特征数量增长
    fig_height = max(6, 2 + num_features * 0.4)
    fig, ax = plt.subplots(figsize=(8, fig_height))  # 保持宽度8，高度动态

    text_size = 8  # 稍微增大了字体 (原为 6)
    spacing = 1.5  # 关键：控制每个特征之间的垂直间距
    text_y_offset = 0.15  # 文本在线条上方的偏移量

    # --- 修改点 结束 ---

    # 定义颜色和形状映射
    palette = {'Pathomics': 'blue', 'Spatial': 'red'}
    markers = {'Pathomics': 'o', 'Spatial': 's'}

    # 2. 使用索引作为y轴坐标，更易于控制
    for idx, row in sig_df.iterrows():
        # y_coord 现在基于索引和间距
        y_coord = idx * spacing + 1

        category = row["Category"]
        beta_val = row["beta"]
        color = palette.get(category, 'gray')
        marker = markers.get(category, 'x')

        # 绘制置信区间和数据点
        ax.plot([row["lower"], row["upper"]], [y_coord, y_coord], color=color, linewidth=2)  # 稍微加粗线条
        ax.plot(beta_val, y_coord, marker=marker, color=color, fillstyle='full', markeredgewidth=0,
                markersize=7)  # 稍微增大标记

        # --- 修改点 开始 ---
        # 3. 将文本放置在线条正上方
        ax.text(
            beta_val,  # X 坐标: beta值 (自动居中)
            y_coord + text_y_offset,  # Y 坐标: 线条上方一点
            row["Feature"],  # 文本内容
            ha='center',  # 水平对齐: 居中
            va='bottom',  # 垂直对齐: 底部 (使文本在y坐标之上)
            fontsize=text_size
        )
        # --- 修改点 结束 ---

    # 手动创建图例
    legend_handles = [
        mlines.Line2D([], [], color=palette['Pathomics'], marker=markers['Pathomics'], linestyle='None',
                      label='Pathomics', markersize=7),
        mlines.Line2D([], [], color=palette['Spatial'], marker=markers['Spatial'], linestyle='None', label='Spatial',
                      markersize=7)
    ]
    ax.legend(handles=legend_handles, title="Feature Category")

    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel("Beta (Effect Size)")

    # --- 修改点 开始 ---
    # 隐藏y轴刻度，并设置范围以提供上下边距
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylim(0, (num_features - 1) * spacing + 2)  # 设置Y轴范围，顶部留出空间
    # --- 修改点 结束 ---

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
