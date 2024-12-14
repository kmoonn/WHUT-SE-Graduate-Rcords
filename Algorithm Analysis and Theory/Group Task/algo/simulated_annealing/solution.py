from algo.base import SolutionBase
import random
import math

# 无穷大的边权
INF = 114514

# 初始温度
INITIAL_TEMPERATURE = 1000

# 每次迭代中温度衰减系数
TEMPERATURE_DECAY_FACTOR = 0.995

# 温度阈值，低于此温度时算法停止
THRESHOLD_TEMPERATURE = 1e-5


class Solution(SolutionBase):
    algorithm_id = "simulated_annealing"  # 算法标识名
    algorithm_name = "Simulated Annealing"

    def __init__(self, input: dict, record_int_data: bool = False):
        """
        :param input: 问题输入 dict{n: 景点数量，m: 边数量，edges: [(u, v, weight), (u, v, weight), ...], best_cost: 最优成本, possible_path: 可能的最优路径}
        :。
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

    def solve(self):
        """
        模拟退火主算法

        :return: (最优路径 list，最优成本 int, 中间数据 { <br>
                    "keys":[(数据 key，数据名), ...],  <br>
                    "int_data": { 数据key: 数据 list} <br>
            })
        """
        # 仍然是随机生成初始解
        path = list(range(self.n))
        # 随机生成初始解
        random.shuffle(path)

        # 计算初始成本
        current_cost = self._path_cost(path)

        # 初始温度
        T = INITIAL_TEMPERATURE

        # 记录初始温度
        if self.int_data is not None:
            self.int_data["temps"] = [T]
            self.int_data["costs"] = [current_cost]

        while True:
            # 随机交换两个节点得到一个领域解
            possible_path = path[:]
            i, j = random.sample(range(self.n), 2)
            possible_path[i], possible_path[j] = possible_path[j], possible_path[i]

            # 计算新成本
            new_cost = self._path_cost(possible_path)

            # 成本和之前成本的差值 Δ
            delta = new_cost - current_cost

            if delta < 0:
                # 新成本更小，接受
                path = possible_path
                current_cost = new_cost
            else:
                # 否则按概率 P 来接受
                # -delta/T 肯定是一个负值，因此 P 区间为 (0, 1]
                # 温度 T 越低，差值 delta 会被放的越大，P 越小
                # 此时代表算法越来越不活跃，逐渐不接受更差的解
                P = math.exp(-delta / T)
                if random.random() < P:
                    path = possible_path
                    current_cost = new_cost

            # 温度衰减
            T *= TEMPERATURE_DECAY_FACTOR

            # 记录温度和成本
            if self.int_data is not None:
                self.int_data["temps"].append(T)
                self.int_data["costs"].append(current_cost)

            # 如果温度低于阈值，则停止
            if T < THRESHOLD_TEMPERATURE:
                break

        res_int_data = None

        if self.int_data is not None:
            res_int_data = {
                "keys": [("costs", "Path Cost"), ("temps", "Temperature")],
                "int_data": self.int_data,
            }

        return path, current_cost, res_int_data
