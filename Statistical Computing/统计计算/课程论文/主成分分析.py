import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load your data (replace 'your_data.xlsx' with the actual file path)
df = pd.read_excel(r'D:\OneDrive\Desktop\ING\统计计算课程论文\data.xlsx')

# Select numeric columns for PCA analysis (exclude non-numeric ones)
columns_for_pca = [
    '国民总收入（亿元）', '国内生产总值（亿元）', '居民消费水平(元)',
    '居民人均可支配收入(元)', '交通运输、仓储和邮政业增加值(亿元)', '国内旅游总花费(亿元)'
]
data_numeric = df[columns_for_pca]

# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_numeric)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(data_standardized)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_ * 100  # Convert to percentage
accumulated_variance_ratio = np.cumsum(explained_variance_ratio)  # Cumulative contribution

# Create a summary table for PCA
pca_summary = pd.DataFrame({
    "主成分": [f"主成分{i+1}" for i in range(len(explained_variance_ratio))],
    "特征值": pca.explained_variance_,
    "方差贡献率 (%)": explained_variance_ratio,
    "累计方差贡献率 (%)": accumulated_variance_ratio
})

# Retrieve component loadings (eigenvectors)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"主成分{i+1}" for i in range(len(explained_variance_ratio))],
    index=columns_for_pca
)

# Display results
print("主成分分析结果：")
print(pca_summary)
print("\n主成分载荷：")
print(loadings)


# Construct principal component expressions
for i, component in enumerate(loadings.columns):
    expression = f"主成分{i+1} = "
    terms = [f"{coef:.4f}*{var}" for coef, var in zip(loadings[component], loadings.index)]
    expression += " + ".join(terms)
    print(expression)
