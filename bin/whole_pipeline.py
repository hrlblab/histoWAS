from pyPheWAS.pyPhewasCorev2 import *
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd

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