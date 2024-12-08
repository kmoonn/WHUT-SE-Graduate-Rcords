# _*_ coding : utf-8 _*_
# @Time : 2024/11/2 下午2:53
# @Author : Kmoon_Hs
# @File : main

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 加载MNIST数据集（使用sklearn的手写数字数据集作为示例）
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 数据预处理
X /= 16  # 归一化

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练SVM模型
model = svm.SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 可视化部分结果
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'True: {y_test[i]}, Pred: {y_pred[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()