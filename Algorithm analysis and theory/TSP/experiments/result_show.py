"""
最短路径结果可视化
"""

from TSP.algo.ant_colony import Solution as AntColonySolution
from TSP.algo.random_climbing import Solution as RandomClimbingSolution
from TSP.algo.simulated_annealing import Solution as SimulatedAnnealingSolution
from TSP.utils.map_generator import show


def result_show(problem):
    # 存储中间数据
    int_data = {}

    # 蚁群算法
    ant_colony_answer = AntColonySolution(problem, int_data).solve()
    print(ant_colony_answer)
    show(problem, ant_colony_answer, "Ant Colony")

    # 爬山算法
    random_climbing_answer = RandomClimbingSolution(problem, int_data).solve()
    print(random_climbing_answer)
    show(problem, random_climbing_answer, "Random Climbing")

    # 模拟退火算法
    simulated_annealing_answer = SimulatedAnnealingSolution(problem, int_data).solve()
    print(simulated_annealing_answer)
    show(problem, simulated_annealing_answer, "Simulated Annealing")
