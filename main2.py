import pandas as pd
import numpy as np
import math
import re
import os
import matplotlib.pyplot as plt
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
from statsmodels.stats.multitest import multipletests

# =============== 1. 读取 & 预处理 ===============



# 读取数据
data = pd.read_csv('/home/yangy50/project/pyPheWAS/feature_rename.csv')


outcome_col = 'Interstitial_Fibrosis____'
# id_col="Biopsy_ID_"

# 其余列视为潜在自变量
feature_cols = [c for c in data.columns if c != outcome_col]
# feature_cols = [c for c in feature_cols if c != id_col]


# 收集所有可用特征列：排除 outcome
feature_cols = [c for c in data.columns if c != outcome_col]


# =============== 2. 批量回归（线性模型） ===============

results = []
for feat in feature_cols:
    # 构造公式： InterstitialFibrosis ~ feat
    formula = f"{outcome_col} ~ {feat}"
    try:
        model = glm(formula, data=data, family=Gaussian()).fit()
        pval = model.pvalues[feat]
        beta = model.params[feat]
        conf_int = model.conf_int().loc[feat].values  # 置信区间 [low, high]
        stderr = model.bse[feat]
        logp = -math.log10(pval) if pval > 0 else float('inf')
        note = ""
    except Exception as e:
        # 如果模型报错(可能缺失值等)，记录 NaN
        pval = np.nan
        beta = np.nan
        conf_int = [np.nan, np.nan]
        stderr = np.nan
        logp = np.nan
        note = f"ERROR: {str(e)}"

    results.append({
        "Feature": feat,
        "Feature Name": feat,  # pyphewas 里常有 'Name' 一列
        "note": note,
        "\"-log(p)\"": logp,
        "p-val": pval,
        "beta": beta,
        "Conf-interval beta": f"[{conf_int[0]}, {conf_int[1]}]",
        "std_error": stderr,
        # 下列几列在原版 pyphewas 里是 PheCode 对应的 category/ICD rollup
        # 这里我们没有，就留空或自定义
        "category_string": None,
        "ICD-9": None,
        "ICD-10": None
    })

# 转成 DataFrame
regressions_df = pd.DataFrame(results)
# 按 p 值升序
regressions_df.sort_values(by="p-val", inplace=True, na_position='last')
regressions_df.reset_index(drop=True, inplace=True)


# =============== 3. 多重比较校正 (Bonferroni / FDR) ===============
pvals = regressions_df["p-val"].values
# Bonferroni
num_tests = np.sum(np.isfinite(pvals))  # 有效比较次数
bon_threshold = 0.05 / num_tests if num_tests > 0 else np.nan

# FDR (Benjamini/Hochberg)
reject_fdr, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh', is_sorted=False)
# 这里 multipletests 不要求 pvals 已排序，但 regressions_df 已排序无妨
# pvals_fdr 是校正后的 p 值
# reject_fdr 是 True/False 列表，表示是否在 FDR 下显著
# 若想找出 FDR 阈值，可以取通过检验中最大原始 p 值
fdr_threshold = None
if any(reject_fdr):
    # 通过 FDR 的那部分 p 值
    passed = pvals[reject_fdr]
    fdr_threshold = max(passed)  # 通过检验的 p 值里最大的那个

# 把校正结果写进 DataFrame
regressions_df["pval_fdr"] = pvals_fdr
regressions_df["bon_sig"] = regressions_df["p-val"] < bon_threshold
regressions_df["fdr_sig"] = reject_fdr

# 另存结果表，类似 pyphewas 的 regressions_RegA.csv
regressions_df.to_csv("regressions_mydata.csv", index=False)


# =============== 4. 复制 pyphewas 常见图表 ===============

# 为了尽量还原 pyphewas 的风格，下面分别写三个绘图函数：
# - plot_manhattan()
# - plot_effect_size()
# - plot_volcano()
# 你可以根据自己需求做微调。

def plot_manhattan(df, threshold, title, save_path):
    """
    df: 回归结果表
    threshold: 画一条 -log10(threshold) 线
    title: 图标题
    save_path: 保存文件名
    """
    # x: feature index, y: -log10(p)
    xs = np.arange(len(df))
    ys = df["\"-log(p)\""].values

    plt.figure(figsize=(8,5))
    plt.scatter(xs, ys, color='blue', s=10)
    if threshold is not None and threshold > 0:
        plt.axhline(-math.log10(threshold), color='red', linestyle='--', label='threshold')

    plt.xlabel("Features")
    plt.ylabel("-log10(p)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_effect_size(df, threshold, title, save_path):
    """
    效应量图：x 轴是 beta, y 轴是 feature 排列。
    只画那些 p < threshold 的点（常见做法是只展示显著的）。
    """
    # 先筛选出 p < threshold
    if threshold is None or threshold <= 0:
        # 如果 threshold 不存在，就直接不画任何线
        sig_df = df.copy()
    else:
        sig_df = df[df["p-val"] < threshold].copy()

    # 没有任何显著特征就画空图
    if sig_df.shape[0] == 0:
        plt.figure()
        plt.title(title + "\n(No features passed threshold)")
        plt.savefig(save_path, dpi=300)
        plt.close()
        return

    # 按 beta 排序，或按 p 值排序
    sig_df.sort_values(by="beta", inplace=True)
    sig_df.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(6, 0.3*len(sig_df)+1))
    yvals = np.arange(len(sig_df))  # y 轴坐标
    betas = sig_df["beta"].values

    # 提取置信区间
    ci_str = sig_df["Conf-interval beta"].values
    # ci_str 形如 "[lower, upper]"，需要解析一下
    lower_ci = []
    upper_ci = []
    for c in ci_str:
        # c 是字符串 "[x, y]"
        c = c.strip("[]")
        parts = c.split(",")
        if len(parts) == 2:
            l, u = float(parts[0]), float(parts[1])
        else:
            l, u = np.nan, np.nan
        lower_ci.append(l)
        upper_ci.append(u)

    plt.errorbar(betas, yvals, xerr=[betas - np.array(lower_ci), np.array(upper_ci) - betas],
                 fmt='o', color='blue', ecolor='lightblue', capsize=3)

    # 在点的左或右侧加文字
    for i, row in sig_df.iterrows():
        plt.text(row["beta"], i, row["Feature"], va='center',
                 ha='left' if row["beta"]>0 else 'right', fontsize=8)

    plt.axvline(x=0, color='black', linewidth=1)
    plt.xlabel("Beta (Effect Size)")
    plt.yticks(yvals, [""]*len(sig_df))  # 不在 y 轴显示文字，避免重叠
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_volcano(df, bon_thr, fdr_thr, title, save_path):
    """
    火山图：x 轴是 beta, y 轴是 -log10(p)
    用颜色或标记区分 Bonferroni 显著 / FDR 显著 / 不显著。
    并给显著点加 label。
    """
    plt.figure(figsize=(8,6))
    betas = df["beta"].values
    logp = df["\"-log(p)\""].values
    pvals = df["p-val"].values

    # 三类颜色
    colors = []
    labels = []
    for i in range(len(df)):
        if bon_thr and pvals[i] < bon_thr:
            # Bonferroni 显著
            colors.append("gold")
            labels.append(df["Feature"].iloc[i])
        elif fdr_thr and pvals[i] < fdr_thr:
            # FDR 显著
            colors.append("midnightblue")
            labels.append(df["Feature"].iloc[i])
        else:
            # 不显著
            colors.append("slategray")
            labels.append("")

    plt.scatter(betas, logp, c=colors, s=20)

    # 给显著的点加 label
    for x, y, lab, col in zip(betas, logp, labels, colors):
        if lab != "":
            plt.text(x, y, lab, fontsize=6, rotation=45, va='bottom')

    plt.xlabel("Beta (Effect Size)")
    plt.ylabel("-log10(p)")
    plt.title(title)
    plt.grid(True, axis='both', linestyle=':', alpha=0.5)

    # 画个图例
    import matplotlib.lines as mlines
    line_bon = mlines.Line2D([], [], color='gold', marker='o', linestyle='None', markersize=5, label='BonFerroni')
    line_fdr = mlines.Line2D([], [], color='midnightblue', marker='o', linestyle='None', markersize=5, label='FDR')
    line_insig = mlines.Line2D([], [], color='slategray', marker='o', linestyle='None', markersize=5, label='Insignificant')
    plt.legend(handles=[line_bon, line_fdr, line_insig], loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =============== 5. 生成各图表，与 pyphewas 类似 ===============

# 5.1 Manhattan Plot (Bonferroni)
plot_manhattan(regressions_df, bon_threshold,
               "Manhattan Plot (Bonferroni)", "Manhattan_bon.png")

# 5.2 Manhattan Plot (FDR)
if fdr_threshold is not None:
    plot_manhattan(regressions_df, fdr_threshold,
                   "Manhattan Plot (FDR)", "Manhattan_fdr.png")
else:
    print("No features passed FDR; skip Manhattan_fdr.png")

# 5.3 Effect Size Plot (Bonferroni)
plot_effect_size(regressions_df, bon_threshold,
                 "Effect Size (Bonferroni)", "EffectSize_bon.png")

# 5.4 Effect Size Plot (FDR)
if fdr_threshold is not None:
    plot_effect_size(regressions_df, fdr_threshold,
                     "Effect Size (FDR)", "EffectSize_fdr.png")

# 5.5 Volcano Plot
plot_volcano(regressions_df, bon_threshold, fdr_threshold,
             "Volcano Plot", "Volcano.png")

print("All done! Check the generated CSV and PNG files.")
