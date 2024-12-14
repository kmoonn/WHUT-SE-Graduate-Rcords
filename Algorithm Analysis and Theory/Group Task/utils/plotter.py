import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 展示时离群值最多能是最小值的几倍
OUTLIER_MULTIPLE_THRESHOLD = 2

# 柱状图柱子颜色列表，循环使用
BAR_COLORS = (
    "#B5C0D0",
    "#CCD3CA",
    "#F5E8DD",
    "#92C7CF",
    "#AAD7D9",
    "#E1AFD1",
    "#436850",
)

# 折线图颜色列表
PLOT_COLORS = (
    "#365486",
    "#7D0A0A",
)


class Plotter:
    def __init__(self):
        # 目前使用到了哪个颜色
        self._color_ind = 0

    def _next_bar_color(self):
        """
        返回下一个柱子颜色

        :return: 颜色 Hex Str
        """
        color = BAR_COLORS[self._color_ind]
        self._color_ind = (self._color_ind + 1) % len(BAR_COLORS)
        return color

    def plot(
        self,
        x,
        y: list | tuple[list],
        title,
        x_label,
        y_label: str | tuple[str],
        ignore_outliers=True,
    ):
        """
        绘制折线图表

        :param x: x 轴数据
        :param y: y 轴数据，可以接受一个二元组，同时展示两个折线
        :param title: 标题
        :param x_label: x 轴标签
        :param y_label: y 轴标签，和 y 一样可以接受一个二元组，为两个折线分别加上标签
        :param ignore_outliers: 是否忽略离群值，如果为 true，会不展示过大的 y
        """
        if isinstance(y, tuple):
            if len(y) != 2 or len(y_label) != 2:
                raise ValueError("y 为元组时只能接受一个二元组，同时展示两个折线")
            y1, y2 = y
            y_label_1, y_label_2 = y_label
        else:
            y1, y2 = y, None
            y_label_1, y_label_2 = y_label, None

        # 找出 y1 中的最小值
        minY = min(y1)
        threshold = minY * OUTLIER_MULTIPLE_THRESHOLD

        fig, ax_1 = plt.subplots(figsize=(10, 6))

        ax_1.set_xlabel(x_label)
        ax_1.set_ylabel(y_label_1, color=PLOT_COLORS[0])
        if ignore_outliers:
            ax_1.set_ylim(minY - 10, threshold)
        ax_1.plot(x, y1, color=PLOT_COLORS[0])
        ax_1.tick_params(axis="y", labelcolor=PLOT_COLORS[0])
        if y2 is None:
            ax_1.grid(True)
        else:
            ax_2 = ax_1.twinx()
            ax_2.set_ylabel(y_label_2, color=PLOT_COLORS[1], rotation=270)
            ax_2.plot(x, y2, color=PLOT_COLORS[1])
            ax_2.tick_params(axis="y", labelcolor=PLOT_COLORS[1])

        plt.title(title)
        plt.show()

    def bar(self, x, y, title, x_label, y_label):
        """
        绘制柱状图

        :param x: x 轴数据（每个柱子的标签）
        :param y: y 轴数据（值）
        :param title: 标题
        :param x_label: x 轴标签
        :param y_label: y 轴标签
        """
        plt.figure(figsize=(10, 6))
        plt.bar(x, y, color=[self._next_bar_color() for _ in range(len(x))])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # 显示每个柱状图上的值
        for i, v in enumerate(y):
            plt.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)

        plt.tight_layout()
        plt.show()
