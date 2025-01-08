![[Pasted image 20241102145208.png]]
### 经典分类算法：支持向量机（SVM）

#### 基本思想
支持向量机（SVM）是一种监督学习算法，主要用于分类问题。SVM的基本思想是通过找到一个最佳的超平面（hyperplane）来划分不同类别的数据点，使得类别之间的间隔（margin）最大化。SVM的核心概念包括：

1. **决策边界**：SVM试图在特征空间中找到一个决策边界，使得不同类别的数据点被最大化的间隔分开。
2. **支持向量**：支持向量是位于决策边界附近的点，它们对决策边界的构建有直接影响。
3. **核函数**：SVM能够通过核函数（如线性核、多项式核、RBF核）将数据映射到高维空间，从而处理非线性可分的数据。

#### 适用场景
- 文本分类（如垃圾邮件检测）。
- 图像分类（如人脸识别）。
- 生物信息学（如基因分类）。
- 任何具有高维特征的数据集，尤其是小样本高维数据（如金融风险评估）。

#### 优缺点

**优点**：
1. **高维空间表现良好**：SVM在高维特征空间中表现优异。
2. **有效处理非线性问题**：通过使用核函数，可以有效处理非线性可分的数据。
3. **健壮性**：对高维数据的过拟合风险相对较低，尤其是在适当选择核函数和正则化参数的情况下。

**缺点**：
1. **计算复杂度高**：对于大规模数据集，训练时间和内存消耗可能会很高。
2. **选择合适的核函数困难**：需要根据具体数据选择合适的核函数和超参数。
3. **不易解释**：相较于决策树等模型，SVM的模型结果不容易进行解释。

### 解决案例：手写数字识别

#### 问题描述
使用SVM算法解决手写数字识别问题，数据集为MNIST数据集。MNIST数据集包含70000张28x28像素的手写数字图像（0到9）。

#### 实验设计

1. **数据预处理**：
   - 将28x28的图像展平为一维数组（784维）。
   - 归一化图像数据，将像素值缩放到[0, 1]区间。
   - 将数据集分为训练集（60000张）和测试集（10000张）。

2. **模型训练**：
   - 使用`sklearn`库中的SVM实现，选择RBF核函数。
   - 训练模型并使用交叉验证来优化超参数（如C和γ值）。

3. **模型评估**：
   - 在测试集上评估模型性能，计算准确率、混淆矩阵和分类报告（精确率、召回率、F1-score）。

4. **结果可视化**：
   - 可视化部分分类结果和模型在测试集上的混淆矩阵。

#### 实现代码示例

```python
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
```
### 实验结果
评估指标
![[Pasted image 20241102150054.png]]
混淆矩阵：
![[Pasted image 20241102150118.png]]
示例：
![[Pasted image 20241102150036.png]]
### 总结
支持向量机是一种强大的分类算法，适合用于多种任务，特别是在高维数据和复杂分类问题中。通过这个手写数字识别的案例，我们可以看到SVM在实际应用中的效果和性能。