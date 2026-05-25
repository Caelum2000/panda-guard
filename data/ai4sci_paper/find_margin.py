import pandas as pd
from collections import defaultdict

# ====== 配置 ======
input_csv = "data/ai4sci_paper/20260423.csv"

# 根据你的CSV实际列名进行修改
subject_col = "Subject"          # 学科列
subsubject_col = "Sub-discipline"   # 子学科列
risk_col = "Risk Dimension"      # 风险维度列

# 输出文件名
subject_mapping_csv = "subject_to_subsubjects.csv"
risk_mapping_csv = "risk_to_subjects.csv"

# ====== 读取数据 ======
df = pd.read_csv(input_csv)

# 去除空值
df = df[[subject_col, subsubject_col, risk_col]].dropna()

# =========================================================
# 1. 每个学科 -> 包含的子学科
# =========================================================
subject_to_subsubjects = (
    df.groupby(subject_col)[subsubject_col]
    .unique()
    .reset_index()
)

# 将 ndarray 转成字符串
subject_to_subsubjects[subsubject_col] = (
    subject_to_subsubjects[subsubject_col]
    .apply(lambda x: ", ".join(sorted(map(str, x))))
)

# 保存
subject_to_subsubjects.to_csv(subject_mapping_csv, index=False)

print(f"Saved: {subject_mapping_csv}")

# =========================================================
# 2. 每个风险维度 -> 对应的学科
# =========================================================
risk_to_subjects = (
    df.groupby(risk_col)[subject_col]
    .unique()
    .reset_index()
)

risk_to_subjects[subject_col] = (
    risk_to_subjects[subject_col]
    .apply(lambda x: ", ".join(sorted(map(str, x))))
)

# 保存
risk_to_subjects.to_csv(risk_mapping_csv, index=False)

print(f"Saved: {risk_mapping_csv}")