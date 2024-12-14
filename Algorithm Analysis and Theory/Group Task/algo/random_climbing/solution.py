from algo.base import SolutionBase
import random
import math

# 无穷大的边权
INF = 114514
# 💡 持续多少轮最优解没有变化就不再进行下去
MAX_NO_IMPROVE_EPOCHS = 1000
# 重启多少次最优解没有变化就不再继续
MAX_NO_IMPROVE_RESTARTS = 3


class Solution(SolutionBase):
    algorithm_id = "random_climbing"  # 算法标识名
    algorithm_name = "Random Climbing"

    def __init__(self, input: dict, record_int_data: bool = False):
        """

        :param input: 问题输入 dict{n: 景点数量，m: 边数量，edges: [(u, v, weight), (u, v, weight), ...], best_cost: 最优成本, possible_path: 可能的最优路径}
        :param record_int_data: 是否记录中间数据
        """
        self.n = input["n"]
        self.m = input["m"]
        # 为了方便处理，用邻接矩阵存储无向图
        # 💡 因为问题可能是非完全图，对于没有指定权重的边，当作权重是 INF 的边
        self.G = [([INF] * self.n) for _ in range(self.n)]
        for u, v, weight in input["edges"]:
            self.G[u][v] = weight
            self.G[v][u] = weight

        self.int_data = {} if record_int_data else None

    def _path_cost(self, path: list) -> int:
        """
        计算路径 path 的总成本
        """
        cost = 0
        for i in range(len(path)):
            cost += self.G[path[i]][path[(i + 1) % len(path)]]
        return cost

    def _solve_once(self) -> tuple[list[int], int]:
        """
        :return: (最优路径 list，最优成本 int)
        """
        # 💡 剩下还能运行多少轮，如果有新的优解出现则继续
        iters_left = MAX_NO_IMPROVE_EPOCHS

        path = list(range(self.n))
        # 随机生成初始解
        random.shuffle(path)

        # 初始成本
        current_cost = self._path_cost(path)

        # 每次重启后清空记录的数据
        if self.int_data is not None:
            self.int_data["costs"] = [current_cost]

        while iters_left > 0:
            # 随机交换两个节点，得到一个邻域解
            possible_path = path[:]
            i, j = random.sample(range(self.n), 2)
            possible_path[i], possible_path[j] = possible_path[j], possible_path[i]

            # 计算新成本
            new_cost = self._path_cost(possible_path)

            # 如果新成本更小，则更新当前最优解
            if new_cost < current_cost:
                current_cost = new_cost
                path = possible_path
                # 剩下还能运行多少轮
                iters_left = MAX_NO_IMPROVE_EPOCHS
            else:
                iters_left -= 1

            # 记录当前成本
            if self.int_data is not None:
                self.int_data["costs"].append(current_cost)

        return path, current_cost

    def solve(self) -> tuple[list[int], int]:
        """
        实现重启爬山算法
        :return: (最优路径 list，最优成本 int, 中间数据 { <br>
                    "keys":[(数据 key，数据名), ...],  <br>
                    "int_data": { 数据key: 数据 list} <br>
            })
        """
        best_path = []
        best_cost = INF

        # 剩下还要重启多少趟
        restart_left = MAX_NO_IMPROVE_RESTARTS
        while restart_left > 0:
            path, cost = self._solve_once()
            if cost < best_cost:
                best_path = path
                best_cost = cost
                # 通过重启能找到新的最优解，那就再多重启几次
                restart_left = MAX_NO_IMPROVE_RESTARTS
            else:
                restart_left -= 1

        res_int_data = None

        if self.int_data is not None:
            res_int_data = {"keys": [("costs", "Path Cost")], "int_data": self.int_data}

        return best_path, best_cost, res_int_data
