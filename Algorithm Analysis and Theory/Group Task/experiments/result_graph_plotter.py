from algo.base import SolutionBase
from utils.map_generator import show as show_map


class ResultGraphPlotter:
    def __init__(self, algorithm_classes: tuple):
        """
        为每个算法的结果绘制图

        :param algorithm_classes: 算法类元组
        :return: self
        """
        self.algorithm_classes = algorithm_classes

    def plot(self, problem):
        """
        在 problem 上运行每个算法一次，生成结果图

        :param problem: TSP 问题
        :return: self
        """
        # 算法实例初始化

        self.algorithms: list[SolutionBase] = [
            algo_class(problem) for algo_class in self.algorithm_classes
        ]

        print(f"[Result Graph Plotter] Running algorithms...")
        for i, algo in enumerate(self.algorithms):
            print(f"Running: {i + 1}/{len(self.algorithms)}")
            answer = algo.solve()
            show_map(problem, answer, algo.algorithm_name)
        print()
        return self
