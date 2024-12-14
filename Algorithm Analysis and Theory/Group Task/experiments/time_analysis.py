"""
时间分析
"""

import numpy as np
from algo.base import SolutionBase
from utils.plotter import Plotter

from utils import Timer


_plotter = Plotter()
_timer = Timer()


class TimeAnalysis:
    def __init__(self, test_count, algorithm_classes: tuple):
        """
        分析算法平均耗时

        :param test_count: 在**每个**问题 problem 下算法的重复执行次数
        :param algorithm_classes: 算法类元组
        """
        self.test_count = test_count
        self.algorithm_classes = algorithm_classes

        # 累积各个算法的耗时
        self.accumulated_times = [0] * len(algorithm_classes)

        # 记录 run 方法被调用了多少次
        self.run_count = 0

        # 初始化最终的平均耗时结果
        self.avg_times = [0] * len(algorithm_classes)

        # 算法实例（待初始化）
        self.algorithms: list[SolutionBase] = []

    def _run_once(self):

        for i, algo in enumerate(self.algorithms):
            _timer.start()
            answer = algo.solve()
            _timer.stop()
            # print(answer)
            # print(f"Time consumed: {timer.time_consumed_ms():.2f}ms")
            # print(timer.time_consumed_ms())
            self.accumulated_times[i] += _timer.time_consumed_ms()

    def run(self, problem):
        """
        重复运行算法 test_count 次

        :param problem: TSP 问题
        :return: self
        """
        # 算法实例初始化
        self.problem = problem
        self.algorithms: list[SolutionBase] = [
            algo_class(problem) for algo_class in self.algorithm_classes
        ]
        self.run_count += 1

        print(f"[Time Analysis] Running algorithms for {self.test_count} times...")
        for iter in range(self.test_count):
            if (iter + 1) % 5 == 0:
                print(f"Running: {iter + 1}/{self.test_count}", flush=True, end="\r")
            self._run_once()
        print()
        return self

    def collect(self):
        """
        计算每个算法的平均耗时

        :return: self
        """
        for i, time in enumerate(self.accumulated_times):
            # 相当于跑了 test_count*run_count 次
            self.avg_times[i] = time / (self.run_count * self.test_count)
        return self

    def show(self):
        # 算法名
        algo_names = [algo.algorithm_name for algo in self.algorithms]

        _plotter.bar(
            algo_names,
            self.avg_times,
            title="Average Time of Three Algorithms",
            x_label="Algorithms",
            y_label="Average Time (ms)",
        )

        return self
