import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load your data (replace 'your_data.xlsx' with your actual file path)
df = pd.read_excel(r'D:\OneDrive\Desktop\ING\统计计算课程论文\data.xlsx')

# Define independent variables (X) and dependent variable (y)
columns_for_regression = [
    '国民总收入（亿元）', '国内生产总值（亿元）'
]
X = df[columns_for_regression]
y = df['国内旅游总花费(亿元)']

# Fit regression model
model = LinearRegression()
model.fit(X, y)

# Retrieve regression equation and R-squared value
coefficients = model.coef_
intercept = model.intercept_
r_squared = model.score(X, y)

# Display the regression equation
print("回归方程:")
print(f"国内旅游总花费 = {intercept:.2f} " + " ".join(
    [f"+ {coeff:.2f}*{col}" for coeff, col in zip(coefficients, X.columns)]
))
print(f"R-squared: {r_squared:.2f}")