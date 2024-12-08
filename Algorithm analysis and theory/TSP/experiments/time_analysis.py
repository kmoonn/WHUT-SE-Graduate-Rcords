"""
时间分析
"""
import numpy as np
from matplotlib import pyplot as plt

from TSP.algo.ant_colony import Solution as AntColonySolution
from TSP.algo.random_climbing import Solution as RandomClimbingSolution
from TSP.algo.simulated_annealing import Solution as SimulatedAnnealingSolution
from TSP.utils import Timer


class TimeAnalysis:
    def __init__(self, problem, numbers):
        self.ant_colony_time = []
        self.random_climbing_time = []
        self.simulated_annealing_time = []
        self.problem = problem
        self.numbers = numbers

    def run(self):
        # 存储中间数据
        int_data = {}

        timer = Timer()

        # 蚁群算法
        timer.start()
        ant_colony_answer = AntColonySolution(self.problem, int_data).solve()
        timer.stop()
        print(ant_colony_answer)
        print(f"Time consumed: {timer.time_consumed_ms():.2f}ms")
        print(timer.time_consumed_ms())
        self.ant_colony_time.append(timer.time_consumed_ms())

        # 爬山算法
        timer.start()
        random_climbing_answer = RandomClimbingSolution(self.problem, int_data).solve()
        print(random_climbing_answer)
        timer.stop()
        print(ant_colony_answer)
        print(f"Time consumed: {timer.time_consumed_ms():.2f}ms")
        self.random_climbing_time.append(timer.time_consumed_ms())

        # 模拟退火算法
        timer.start()
        simulated_annealing_answer = SimulatedAnnealingSolution(self.problem, int_data).solve()
        print(simulated_annealing_answer)
        timer.stop()
        print(ant_colony_answer)
        print(f"Time consumed: {timer.time_consumed_ms():.2f}ms")
        self.simulated_annealing_time.append(timer.time_consumed_ms())

    def show(self):
        for i in range(self.numbers):
            self.run()
        algorithms = ["ant_colony", "random_climbing", "simulated_annealing"]
        avg_times = [np.mean(self.ant_colony_time),
                     np.mean(self.random_climbing_time),
                     np.mean(self.simulated_annealing_time)]
        # 绘制直方图
        print(avg_times)
        plt.bar(algorithms, avg_times, color=['blue', 'orange', 'green'], alpha=0.7)
        plt.xlabel('Algorithms')
        plt.ylabel('Average Running Time (ms)')
        plt.title('Average Running Time of Three Algorithms')

        # 显示每个柱状图上的平均值
        for i, v in enumerate(avg_times):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

        # 显示图表
        plt.tight_layout()
        plt.show()
