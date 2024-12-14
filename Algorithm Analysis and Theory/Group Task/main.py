"""
实验程序总入口
"""

from reader import read_question

from algo.ant_colony import Solution as AntColonySolution
from algo.random_climbing import Solution as RandomClimbingSolution
from algo.simulated_annealing import Solution as SimulatedAnnealingSolution


from experiments import IterationAnalysis
from experiments import TimeAnalysis
from experiments import AccuracyAnalysis
from experiments import ResultGraphPlotter

if __name__ == "__main__":
    problem_1 = read_question("questions/question_density_100_3.md")
    problem_2 = read_question("questions/question_density_100_4.md")

    # 中间迭代过程
    IterationAnalysis(
        algorithm_classes=(
            AntColonySolution,
            RandomClimbingSolution,
            SimulatedAnnealingSolution,
        ),
    ).run(problem_1)
    # 绘制解出的最短路径结果图
    ResultGraphPlotter(
        algorithm_classes=(
            AntColonySolution,
            RandomClimbingSolution,
            SimulatedAnnealingSolution,
        ),
    ).plot(problem_1)
    # 平均时间直方图
    TimeAnalysis(
        test_count=100,
        algorithm_classes=(
            AntColonySolution,
            RandomClimbingSolution,
            SimulatedAnnealingSolution,
        ),
    ).run(problem_1).run(problem_2).collect().show()
    # 准确率直方图
    AccuracyAnalysis(
        test_count=100,
        algorithm_classes=(
            AntColonySolution,
            RandomClimbingSolution,
            SimulatedAnnealingSolution,
        ),
    ).run(problem_1).run(problem_2).collect().show()

    # 从稀疏到稠密，算法平均耗时折线图（衡量稳定性）
    
    # 从稀疏到稠密，算法准确率折线图（衡量稳定性）
