import matplotlib.pyplot as plt

# 展示时离群值最多能是最小值的几倍
OUTLIER_MULTIPLE_THRESHOLD = 3


class Plotter:
    def __init__(self):
        pass

    def plot(self, x, y, title, x_label, y_label, ignore_outliers=True):
        """
        绘制图表

        :param x: x 轴数据
        :param y: y 轴数据
        :param title: 标题
        :param x_label: x 轴标签
        :param y_label: y 轴标签
        :param ignore_outliers: 是否忽略离群值，如果为 true，会不展示过大的 y
        """
        minY = min(y)
        threshold = minY * OUTLIER_MULTIPLE_THRESHOLD
        plt.figure(figsize=(10, 6))
        if ignore_outliers:
            plt.ylim(minY - 10, threshold)
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.show()
