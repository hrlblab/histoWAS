import numpy as np
import pandas as pd
from pyPheWAS.pyPhewasCorev2 import *


# 读取数据
data = pd.read_csv('/home/yangy50/project/pyPheWAS/feature_rename.csv')


outcome_col = 'Interstitial_Fibrosis____'
# id_col="Biopsy_ID_"

# 其余列视为潜在自变量
feature_cols = [c for c in data.columns if c != outcome_col]
# feature_cols = [c for c in feature_cols if c != id_col]


# 收集所有可用特征列：排除 outcome
feature_cols = [c for c in data.columns if c != outcome_col]


# data: 已处理后的 DataFrame
# outcome_col: 表型列（例如 "Interstitial_Fibrosis"）
# feature_cols: 除 outcome_col 外的其它列名
N = data.shape[0]
P = len(feature_cols)

# fm[0] 为自变量矩阵：shape (N, P)
fm0 = data[feature_cols].to_numpy()
# fm[1] 与 fm[2] 均设为零矩阵
fm1 = np.zeros((N, P))
fm2 = np.zeros((N, P))
fm = np.array([fm0, fm1, fm2])

# demo DataFrame 包含表型数据
demo = data[[outcome_col]].copy()
# 如有其他协变量，也可加入，比如： demo['age'] = data['age']


def run_featurewas(fm, demo, feature_names, reg_type, covariates='', target='Interstitial_Fibrosis', phe_thresh=5,
                   canonical=False, force_linear=True):
    """
    批量回归分析，每个特征作为自变量，对目标表型（target）进行线性回归。

    参数说明：
      fm: 3×N×P 的特征矩阵，其中 fm[0] 为各特征数据（N:样本数，P:特征数），fm[1]与 fm[2] 可置零。
      demo: 包含目标变量和其它协变量的 DataFrame，必须包含 target 列（例如 'Interstitial_Fibrosis'）。
      feature_names: 长度为 P 的列表，每个元素为对应特征的名称（例如 ["A1", "B2", "C3", ...]）。
      reg_type: 回归类型参数（这里不再实际使用，可传入 1 表示线性）。
      covariates: 可选的协变量（例如 'age+sex'），如果为空则只用目标变量。
      target: 目标变量名称，作为因变量（例如 'Interstitial_Fibrosis'）。
      phe_thresh: 进行回归的最低阈值（可参考原版用法）。
      canonical: 固定 False，使公式为 target ~ phe + covariates。
      force_linear: 如果为 True，则强制使用线性回归。

    返回：
      regressions: 一个 DataFrame，包含每个特征对应的回归结果（p 值、beta、置信区间等）。
      model_str: 构造的回归模型公式字符串。

    说明：
      该函数改造自 pyphewas 的 run_phewas 函数，但去除了 ICD/CPT 相关内容，直接对每个特征（来自 fm[0]）与目标变量做回归分析，
      模型公式为 "target ~ phe [+ covariates]"，其中每次将 fm[0] 的一列赋值给临时变量 "phe"。
    """
    import logging
    from tqdm import tqdm
    import numpy as np
    import pandas as pd

    log = logging.getLogger(__name__)

    # 如果 demo 中有 id 列，则按 id 排序（可选）
    if 'id' in demo.columns:
        log.info('Sorting demo data by id...')
        demo.sort_values(by='id', inplace=True)
        demo.reset_index(inplace=True, drop=True)

    # fm[0] 的形状：(N, P)
    P = fm[0].shape[1]

    # 定义模型公式。对于 canonical=False，我们希望构造：
    #    dependent = target, independents = ["phe"] + (其他协变量)
    independents = covariates.split('+') if covariates != '' else []
    independents.insert(0, "phe")  # 第一个自变量为临时变量 "phe"（即当前特征值）
    dependent = target
    model_str = dependent + '~' + "+".join(independents)

    # 强制使用线性回归
    model_type = "linear" if force_linear else ("linear" if canonical and (reg_type != 0) else "log")

    # 准备模型数据：demo 必须包含目标变量以及所有协变量（不包含 "phe"，因为我们在循环中动态赋值）
    cov_list = [cov.strip() for cov in covariates.split('+')] if covariates != '' else []
    cols = [target] + cov_list
    model_data = demo[cols].copy()

    # 定义输出结果的 DataFrame，其格式参考 pyphewas 的 regressions 输出
    reg_columns = ["Feature", "Feature Name", "note", "\"-log(p)\"", "p-val", "beta", "Conf-interval beta", "std_error"]
    regressions = pd.DataFrame(columns=reg_columns)

    # 循环遍历每个特征（共 P 个）
    for index in tqdm(range(P), desc='Running Regressions'):
        # 使用 feature_names 中的名称作为该特征的信息
        phe_info = [feature_names[index], feature_names[index]]

        # 将当前特征（fm[0] 的第 index 列）赋值到模型数据的 'phe' 列中
        model_data['phe'] = fm[0][:, index]

        # 调用 pyphewas 中的 fit_pheno_model 来拟合模型（该函数应已定义）
        phe_model, note = fit_pheno_model(model_str, model_type, model_data, phe_thresh=phe_thresh)

        # 调用 pyphewas 中的 parse_pheno_model 将模型结果解析并追加到 regressions DataFrame 中
        # 这里 independents[0] 应为 "phe"
        parse_pheno_model(regressions, phe_model, note, phe_info, independents[0])

        # 重置 'phe' 列，确保下次循环时干净
        model_data['phe'] = np.nan

    # 按 p 值排序后返回回归结果和模型公式
    return regressions.sort_values(by='p-val'), model_str


regressions_df, model_equation = run_featurewas(fm, demo, feature_cols, reg_type=1,
                                                 covariates="", target=outcome_col,
                                                 phe_thresh=5, canonical=False, force_linear=True)


def get_bon_thresh(p_values, alpha=0.05):
    # p_values 是一个数组或列表，包含了所有可用（非NaN）的p值
    # 先数一数有多少个有效p值
    num_tests = np.sum(np.isfinite(p_values))
    if num_tests == 0:
        return np.nan
    return alpha / num_tests


def get_fdr_thresh(p_values, alpha=0.05):
    # 排序所有可用p值
    sn = np.sort(p_values[np.isfinite(p_values)])
    if len(sn) == 0:
        return np.nan

    # 遍历 sorted p-values
    # 当 p[i] <= (i+1)/M * alpha 时，认为是FDR显著
    # 这里 i+1 是从1开始计数
    M = len(sn)
    fdr_thresh = np.nan
    for i, pval in enumerate(sn):
        crit = alpha * (i + 1) / M
        if pval <= crit:
            fdr_thresh = pval
        else:
            break
    return fdr_thresh


pvals = regressions_df["p-val"].values

# 计算 Bonferroni 阈值
bon_threshold = get_bon_thresh(pvals, alpha=0.05)

# 计算 FDR 阈值
fdr_threshold = get_fdr_thresh(pvals, alpha=0.05)

# 如果想用某个统一阈值，比如0.05
threshold_value = 0.05


def plot_manhattan_feature(regressions, *, thresh, save='', save_format='png'):
    """
    绘制特征回归结果的曼哈顿图，不依赖 ICD/CPT 映射信息。

    参数:
      regressions: 回归结果 DataFrame，需包含列 "\"-log(p)\"" 和 "p-val"。
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
    # 画阈值线
    ax.axhline(y=-math.log10(thresh), color='red', linestyle='--', label='Threshold (-log10)')

    ax.set_xlabel("Feature Index")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Manhattan Plot (Features)")
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, format=save_format, dpi=300)
        plt.close()
    else:
        plt.show()


# 调用 plot_manhattan 时必须使用关键字参数 'thresh'
plot_manhattan_feature(regressions_df, thresh=threshold_value, save="Manhattan_mydata.png", save_format="png")


def plot_effect_size_feature(regressions, *, thresh, save='', save_format='png'):
    """
    绘制特征效应量图：横轴为回归系数（beta），误差条表示置信区间，
    仅显示 p 值小于 thresh 的显著特征。

    参数:
      regressions: 回归结果 DataFrame，需包含列 "Feature", "\"-log(p)\"", "p-val", "beta", "Conf-interval beta"
      thresh: 显著性 p 值阈值（例如 Bonferroni 阈值），只有 p-val < thresh 的特征会被绘制
      save: 如果提供文件名，则保存图像；否则显示图像
      save_format: 保存图像的格式（如 "png"）
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 筛选出 p 值小于 thresh 的特征
    sig_df = regressions[regressions["p-val"] < thresh].copy()
    if sig_df.empty:
        print("没有特征通过阈值筛选")
        return

    # 按 beta 排序，便于图形展示
    sig_df.sort_values(by="beta", inplace=True)
    sig_df.reset_index(drop=True, inplace=True)

    # 解析置信区间
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

    # 计算左右误差
    beta_vals = sig_df["beta"].values
    error_left = beta_vals - sig_df["lower"].values
    error_right = sig_df["upper"].values - beta_vals

    # 使用特征数目决定图形高度
    y_positions = np.arange(len(sig_df))

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(sig_df) + 2))
    ax.errorbar(beta_vals, y_positions, xerr=[error_left, error_right],
                fmt='o', color='blue', ecolor='lightblue', capsize=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sig_df["Feature"], fontsize=8)
    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel("Beta (Effect Size)")
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

plot_effect_size_feature(regressions_df, thresh=threshold_value, save="EffectSize_mydata.png", save_format="png")
plot_volcano_feature(regressions_df, save="Volcano_mydata.png", save_format="png")