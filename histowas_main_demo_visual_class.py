from bin.whole_pipeline import *
from visualize.histowas_plot_category import *
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd



# 读取数据
data = pd.read_csv('/media/yangy50/Elements/KPMP_new/merged_features.csv')
data.drop(columns=['Biopsy ID: '], inplace=True)
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



# 定义Spatial特征的关键词
spatial_keywords = ['ANN', 'g_function', 'K_function', 'L_function']

# 定义一个函数来判断特征类别
def get_feature_category(feature_name):
    if any(keyword in feature_name for keyword in spatial_keywords):
        return 'Spatial'
    return 'Pathomics'

# 为regressions_df添加一个新的'Category'列
# 注意：regressions_df的索引应该就是特征名，如果不是，请根据实际情况调整
regressions_df['Category'] = regressions_df['Feature'].map(get_feature_category)


pvals = regressions_df["p-val"].values

# 计算 Bonferroni 阈值
bon_threshold = get_bon_thresh(pvals, alpha=0.05)

# 计算 FDR 阈值
fdr_threshold = get_fdr_thresh(pvals, alpha=0.05)

# 如果想用某个统一阈值，比如0.05
threshold_value = 0.05


# 调用 plot_manhattan 时必须使用关键字参数 'thresh'
# 调用新的 plot_manhattan_with_categories
plot_manhattan_feature_categorized(regressions_df,
                               thresh=threshold_value,
                               save="result/Manhattan_demo_topology_category.png",
                               save_format="png")

# 调用新的 plot_effect_size_with_categories
plot_effect_size_feature_categorized(regressions_df,
                                 thresh=threshold_value,
                                 save="result/EffectSize_demo_topology_category.png",
                                 save_format="png")

# 调用新的 plot_volcano_with_categories
plot_volcano_feature_categorized(regressions_df,
                             save="result/Volcano_demo_topology_category.png",
                             save_format="png")
