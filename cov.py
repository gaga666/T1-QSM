from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel('../PD_cluster/UQ.xlsx')
df = df.iloc[:, [2,3,5]]
# 初始化一个字典来存储每个类别变量的编码器
label_encoders = {}

# 对每一列进行标签编码
for column in df.columns:
    if df[column].dtype == 'object':  # 检查是否是字符串或类别数据
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])  # 将字符串编码为整数
        label_encoders[column] = le  # 保存编码器以便以后使用

# 将数据转换为NumPy数组
covariates = df.values
covariates_array = np.array(covariates, dtype=np.float32)

# 打印检查
print(covariates_array)



