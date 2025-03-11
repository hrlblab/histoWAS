import pandas as pd
import numpy as np
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
import math
import matplotlib.pyplot as plt

import re

def make_safe_colname(col):
    # 用正则把非字母数字下划线的字符都替换成下划线
    # 同时去掉首尾的空白
    col = col.strip()
    return re.sub(r'[^0-9a-zA-Z_]', '_', col)



data = pd.read_csv('/home/yangy50/project/pyPheWAS/features.csv')

safe_cols = {}
for c in data.columns:
    safe_cols[c] = make_safe_colname(c)

data.rename(columns=safe_cols, inplace=True)

outcome_col = make_safe_colname('Interstitial Fibrosis (%)')
id_col=make_safe_colname("Biopsy ID: ")

# 其余列视为潜在自变量
feature_cols = [c for c in data.columns if c != outcome_col]
feature_cols = [c for c in feature_cols if c != id_col]


results = []
for feat in feature_cols:
    # 构造公式： InterstitialFibrosis ~ feat
    formula = f"{outcome_col} ~ {feat}"

    # 拟合线性回归
    try:
        model = glm(formula, data=data, family=Gaussian()).fit()
        pval = model.pvalues[feat]
        beta = model.params[feat]
        conf_int = model.conf_int().loc[feat].values  # 置信区间 [low, high]
        stderr = model.bse[feat]
        logp = -math.log10(pval) if pval>0 else float('inf')
        note = ""
    except Exception as e:
        # 如果模型报错(可能是缺失值或其他问题)，则跳过或记录
        pval = np.nan
        beta = np.nan
        conf_int = [np.nan, np.nan]
        stderr = np.nan
        logp = np.nan
        note = f"ERROR: {str(e)}"

    results.append({
        "Feature": feat,
        "beta": beta,
        "pval": pval,
        "-log10(p)": logp,
        "ConfInt": f"[{conf_int[0]}, {conf_int[1]}]",
        "StdErr": stderr,
        "note": note
    })

results_df = pd.DataFrame(results)
# 按p值升序排个序
results_df.sort_values(by="pval", inplace=True)



from statsmodels.stats.multitest import multipletests

pvals = results_df['pval'].values
# Bonferroni
bon_threshold = 0.05 / np.sum(np.isfinite(pvals))
# FDR
reject_fdr, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
fdr_threshold = max(pvals[reject_fdr]) if any(reject_fdr) else None

results_df['bon_sig'] = results_df['pval'] < bon_threshold
results_df['fdr_sig'] = reject_fdr

def plot_manhattan(results_df, bon_threshold, fdr_threshold=None):
    fig, ax = plt.subplots(figsize=(10,5))
    # x: 0,1,2,3..., y: -log10(p)
    xs = np.arange(len(results_df))
    ys = results_df['-log10(p)'].values
    ax.scatter(xs, ys, color='blue', s=10)

    # 画阈值线 (Bonferroni)
    ax.axhline(-math.log10(bon_threshold), color='red', linestyle='--', label='Bonferroni')

    # 如果有FDR阈值
    if fdr_threshold is not None:
        ax.axhline(-math.log10(fdr_threshold), color='green', linestyle='--', label='FDR')

    ax.set_xlabel('Features')
    ax.set_ylabel('-log10(p)')
    ax.set_title('Manhattan Plot for Features')
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_manhattan(results_df, bon_threshold, fdr_threshold)

