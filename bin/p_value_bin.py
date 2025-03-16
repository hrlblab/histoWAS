import numpy as np


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