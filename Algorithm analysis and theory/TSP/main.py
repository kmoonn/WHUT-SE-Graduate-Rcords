"""
实验程序总入口
"""
from TSP.reader import read_question
from experiments.iterative_process import iterative_process
from experiments.time_analysis import TimeAnalysis
from experiments.result_show import result_show

if __name__ == "__main__":
    problem = read_question("./questions/question_density_100_3.md")

    # 中间迭代过程
    iterative_process(problem)
    # 最短路径结果
    result_show(problem)
    # 平均时间直方图
    TimeAnalysis(problem, numbers=10).show()
