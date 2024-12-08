"""
迭代过程中间结果
"""

from TSP.algo.ant_colony import Solution as AntColonySolution
from TSP.algo.random_climbing import Solution as RandomClimbingSolution
from TSP.algo.simulated_annealing import Solution as SimulatedAnnealingSolution
from TSP.utils import Timer, Plotter


def iterative_process(problem):
    timer = Timer()
    plotter = Plotter()

    # 存储中间数据
    int_data = {}

    # 蚁群算法
    timer.start()
    ant_colony_answer = AntColonySolution(problem, int_data).solve()
    timer.stop()
    print(ant_colony_answer)
    print(f'Best cost: {problem["best_cost"]}')
    print(f"Time consumed: {timer.time_consumed_ms():.2f}ms")

    # 绘制蚁群算法中间数据图表
    plotter.plot(
        list(range(1, len(int_data["costs"]) + 1)),
        int_data["costs"],
        "Ant Colony",
        "Iteration",
        "PathCost",
    )

    # 爬山算法
    timer.start()
    random_climbing_answer = RandomClimbingSolution(problem, int_data).solve()
    timer.stop()
    print(random_climbing_answer)
    print(f'Best cost: {problem["best_cost"]}')
    print(f"Time consumed: {timer.time_consumed_ms():.2f}ms")

    # 绘制爬山算法中间数据图表
    plotter.plot(
        list(range(1, len(int_data["costs"]) + 1)),
        int_data["costs"],
        "Random Climbing",
        "Iteration",
        "Cost",
    )

    # 模拟退火算法
    timer.start()
    simulated_annealing_answer = SimulatedAnnealingSolution(problem, int_data).solve()
    timer.stop()
    print(simulated_annealing_answer)
    print(f'Best cost: {problem["best_cost"]}')
    print(f"Time consumed: {timer.time_consumed_ms():.2f}ms")

    # 绘制模拟退火中间数据图表
    plotter.plot(
        int_data["temps"],
        int_data["costs"],
        "Simulated Annealing",
        "Temperature",
        "Cost",
    )
