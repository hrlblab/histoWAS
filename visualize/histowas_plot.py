# def plot_manhattan_feature(regressions, *, thresh, save='', save_format='png'):
#     """
#     绘制特征回归结果的曼哈顿图，不依赖 ICD/CPT 映射信息。
#
#     参数:
#       regressions: 回归结果 DataFrame，需包含列 "\"-log(p)\"" 和 "p-val"。
#       thresh: 用于绘制水平阈值线的 p 值阈值（例如 Bonferroni 阈值）。
#       save: 如果提供文件名，则保存图像；否则直接显示。
#       save_format: 保存的格式（例如 "png"）。
#     """
#     import matplotlib.pyplot as plt
#     import math
#
#     # 使用特征在回归结果中的顺序作为 x 轴
#     xs = list(range(len(regressions)))
#     # y 轴使用 "-log(p)" 数值
#     ys = regressions['"-log(p)"'].values
#
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.scatter(xs, ys, color='blue', s=20)
#     # 画阈值线
#     ax.axhline(y=-math.log10(thresh), color='red', linestyle='--', label='Threshold (-log10)')
#
#     ax.set_xlabel("Feature Index")
#     ax.set_ylabel("-log10(p)")
#     ax.set_title("Manhattan Plot (Features)")
#     ax.legend()
#     plt.tight_layout()
#     if save:
#         plt.savefig(save, format=save_format, dpi=300)
#         plt.close()
#     else:
#         plt.show()


def plot_manhattan_feature(regressions, *, thresh, save='', save_format='png'):
    """
    绘制特征回归结果的曼哈顿图，不依赖 ICD/CPT 映射信息。

    参数:
      regressions: 回归结果 DataFrame，需包含列 "\"-log(p)\""、"p-val" 和 "Feature"。
      thresh: 用于绘制水平阈值线的 p 值阈值（例如 Bonferroni 阈值）。
      save: 如果提供文件名，则保存图像；否则直接显示。
      save_format: 保存的格式（例如 "png"）。
    """
    import matplotlib.pyplot as plt
    import math
    # 使用特征在回归结果中的顺序作为 x 轴
    xs = list(range(len(regressions)))
    # y 轴使用 "-log(p)" 数值
    ys = regressions['"-log(p)"'].values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, color='blue', s=20)
    # 绘制阈值线
    ax.axhline(y=-math.log10(thresh), color='red', linestyle='--', label='Threshold (-log10)')

    # 设置 x 轴标签为特征名称，并旋转 90 度
    ax.set_xticks(xs)
    ax.set_xticklabels(regressions["Feature"], rotation=90)

    ax.set_xlabel("Feature")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Manhattan Plot (Features)")
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, format=save_format, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_effect_size_feature(regressions, *, thresh, save='', save_format='png'):
    """
    绘制特征效应量图：仅显示 p 值小于 thresh 的显著特征，
    使用每个特征的 beta 值及其置信区间（从 "Conf-interval beta" 中解析），
    并以每个点间隔 15 的方式竖直排列（与 pyPheWAS 保持一致）。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 筛选出 p 值小于 thresh 的特征
    sig_df = regressions[regressions["p-val"] < thresh].copy()
    if sig_df.empty:
        print("没有特征通过阈值筛选")
        return

    # 解析置信区间字符串，提取数值
    lower_bounds = []
    upper_bounds = []
    for ci in sig_df["Conf-interval beta"]:
        try:
            ci_str = ci.strip("[]")
            low_str, high_str = ci_str.split(",")
            low = float(low_str.strip())
            high = float(high_str.strip())
        except Exception as e:
            low, high = np.nan, np.nan
        lower_bounds.append(low)
        upper_bounds.append(high)
    sig_df["lower"] = lower_bounds
    sig_df["upper"] = upper_bounds

    # 根据 beta 排序（如果需要，可以按其它方式排序，使图中顺序与 pypheWAS 保持一致）
    sig_df.sort_values(by="beta", inplace=True)
    sig_df.reset_index(drop=True, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_coord = 1  # 初始 y 坐标
    text_size = 6

    for idx, row in sig_df.iterrows():
        beta_val = row["beta"]
        # 根据 beta 正负决定标签水平对齐方式
        if beta_val > 0:
            ax.text(beta_val, y_coord, row["Feature"], ha='left', fontsize=text_size)
        else:
            ax.text(beta_val, y_coord, row["Feature"], ha='right', fontsize=text_size)

        # 绘制数据点
        ax.plot(beta_val, y_coord, 'o', color='xkcd:aqua', fillstyle='full', markeredgewidth=0)
        # 绘制置信区间横线
        ax.plot([row["lower"], row["upper"]], [y_coord, y_coord], color='xkcd:aqua')
        y_coord += 15  # 每个点之间的固定间隔

    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel("Beta (Effect Size)")
    ax.set_yticks([])  # 不显示 y 轴刻度
    ax.set_title("Effect Size Plot (Significant Features)")
    plt.tight_layout()

    if save:
        plt.savefig(save, format=save_format, dpi=300)
        plt.close()
    else:
        plt.show()






def plot_volcano_feature(regressions, *, save='', save_format='png'):
    """
    绘制特征火山图：横轴为回归系数（beta），纵轴为 -log10(p)。
    根据 p 值显著性用不同颜色标记：
      - 如果 p-val < Bonferroni 阈值，用金色 ("gold")
      - 如果 p-val < FDR 阈值，用深蓝 ("midnightblue")
      - 否则用灰色 ("slategray")
    并为显著的点加上特征名称标签。

    参数:
      regressions: 回归结果 DataFrame，需包含 "Feature", "\"-log(p)\"", "p-val", "beta"
      save: 如果提供文件名，则保存图像；否则显示图像
      save_format: 图像保存格式（如 "png"）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import matplotlib.lines as mlines

    # 计算阈值
    def get_bon_thresh(p_values, alpha=0.05):
        return alpha / sum(np.isfinite(p_values))

    def get_fdr_thresh(p_values, alpha=0.05):
        sn = np.sort(p_values)
        sn = sn[np.isfinite(sn)]
        for i in range(len(sn)):
            p_crit = alpha * float(i + 1) / float(len(sn))
            if sn[i] <= p_crit:
                continue
            else:
                break
        return sn[i]

    pvals = regressions["p-val"].values
    bon_threshold = get_bon_thresh(pvals, alpha=0.05)
    fdr_threshold = get_fdr_thresh(pvals, alpha=0.05)

    # 火山图：横轴 beta，纵轴 -log10(p)
    beta_vals = regressions["beta"].values
    logp_vals = regressions['"-log(p)"'].values

    colors = []
    labels = []
    for i, p in enumerate(regressions["p-val"]):
        if p < bon_threshold:
            colors.append("gold")
            labels.append(regressions["Feature"].iloc[i])
        elif p < fdr_threshold:
            colors.append("midnightblue")
            labels.append(regressions["Feature"].iloc[i])
        else:
            colors.append("slategray")
            labels.append("")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(beta_vals, logp_vals, c=colors, s=20)

    # 为显著点添加标签
    for i, label in enumerate(labels):
        if label != "":
            ax.text(beta_vals[i], logp_vals[i], label, fontsize=6, rotation=45, va='bottom')

    ax.set_xlabel("Beta (Effect Size)")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Volcano Plot (Features)")
    ax.axvline(x=0, color="black", linestyle="--")

    # 构造图例
    line_bon = mlines.Line2D([], [], color='gold', marker='o', linestyle='None', markersize=5,
                             label='Bonferroni significant')
    line_fdr = mlines.Line2D([], [], color='midnightblue', marker='o', linestyle='None', markersize=5,
                             label='FDR significant')
    line_insig = mlines.Line2D([], [], color='slategray', marker='o', linestyle='None', markersize=5,
                               label='Not significant')
    ax.legend(handles=[line_bon, line_fdr, line_insig])

    plt.tight_layout()
    if save:
        plt.savefig(save, format=save_format, dpi=300)
        plt.close()
    else:
        plt.show()