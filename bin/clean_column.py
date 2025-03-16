import pandas as pd
import re

def make_safe_colname(col):
    col = col.strip()
    # 只保留数字、字母、下划线，其他字符全部转成下划线
    return re.sub(r'[^0-9a-zA-Z_]', '_', col)

# 读取csv文件
df = pd.read_csv("/home/yangy50/project/pyPheWAS/features.csv")

# 对所有列名进行处理
df.columns = [make_safe_colname(col) for col in df.columns]

# 保存修改后的csv文件
df.to_csv("/home/yangy50/project/pyPheWAS/feature_rename.csv", index=False)
