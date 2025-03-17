from bin.whole_pipeline import *
from visualize.histowas_plot import *
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd



# 读取数据
data = pd.read_csv('data/feature_demo.csv')
# 对 DataFrame 的每一列使用 z-score 标准化
data = (data - data.mean()) / data.std()


outcome_col = 'Interstitial_Fibrosis'
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


regressions_df, model_equation = run_featurewas(fm, demo, feature_cols, reg_type=1,
                                                 covariates="", target=outcome_col,
                                                 phe_thresh=5, canonical=False, force_linear=True)


pvals = regressions_df["p-val"].values

# 计算 Bonferroni 阈值
bon_threshold = get_bon_thresh(pvals, alpha=0.05)

# 计算 FDR 阈值
fdr_threshold = get_fdr_thresh(pvals, alpha=0.05)

# 如果想用某个统一阈值，比如0.05
threshold_value = 0.05


# 调用 plot_manhattan 时必须使用关键字参数 'thresh'
plot_manhattan_feature(regressions_df, thresh=threshold_value, save="result/Manhattan_demo.png", save_format="png")


plot_effect_size_feature(regressions_df, thresh=threshold_value, save="result/EffectSize_demo.png", save_format="png")
plot_volcano_feature(regressions_df, save="result/Volcano_demo.png", save_format="png")
