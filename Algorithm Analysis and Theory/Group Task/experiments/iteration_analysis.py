"""
生成迭代中间结果的图表
"""

from algo.base import SolutionBase
from utils.plotter import Plotter


_plotter = Plotter()


class IterationAnalysis:
    def __init__(self, algorithm_classes: tuple):
        """
        准备生成迭代中间结果的图表

        :param algorithm_classes: 算法类元组
        """
        self.algorithm_classes = algorithm_classes

    def run(self, problem):
        """
        在 problem 上运行每个算法一次，生成迭代中间结果图表

        :param problem: TSP 问题
        :return: self
        """
        # 算法实例初始化

        self.algorithms: list[SolutionBase] = [
            algo_class(problem, True) for algo_class in self.algorithm_classes
        ]

        print(f"[Iteration Analysis] Running algorithms...")
        for i, algo in enumerate(self.algorithms):
            print(f"Running: {i + 1}/{len(self.algorithms)}", flush=True, end="\r")
            _, _, res_int_data = algo.solve()
            int_data_keys = res_int_data["keys"]
            int_data = res_int_data["int_data"]
            # 横坐标
            x_iters = list(range(1, len(int_data["costs"]) + 1))
            first_key, first_label = int_data_keys[0]
            y = int_data[first_key]
            y_label = first_label
            if len(int_data_keys) > 1:
                second_key, second_label = int_data_keys[1]
                y = (y, int_data[second_key])
                y_label = (y_label, second_label)

            # 绘图
            _plotter.plot(
                x_iters,
                y,
                algo.algorithm_name,
                "Iteration",
                y_label,
            )
        print()

        return self
