import random

# 无穷大的边权，代表不可达
INF_DIST = 9999999

# 蚂蚁数量
NUM_ANTS = 10

# 迭代次数
NUM_ITERATIONS = 10

# 信息素的重要性
ALPHA = 1

# 启发式信息的重要性
BETA = 2

# 信息素挥发（衰减）率
EVAPORATION_RATE = 0.5

# 信息素强度，在更新信息素时使用
Q = 100


class Solution:
    def __init__(self, input: dict, int_data: dict | None = None):
        """
        :param input: 问题输入 dict{n: 景点数量，m: 边数量，edges: [(u, v, weight), (u, v, weight), ...], best_cost: 最优成本, possible_path: 可能的最优路径}
        :param int_data: 存储中间数据的字典，会被原地修改。字段包括计算得到的path_cost。没有则不记录。
        """
        self.n = input["n"]
        self.m = input["m"]

        # 生成一个邻接矩阵，存储每两个顶点间的距离
        self.G = [([INF_DIST] * self.n) for _ in range(self.n)]
        # 自己到自己距离为 0
        for i in range(self.n):
            self.G[i][i] = 0

        for u, v, dist in input["edges"]:
            self.G[u][v] = dist
            self.G[v][u] = dist

        # 初始化信息素矩阵 P，最初信息素全为 1
        self.P = [([1] * self.n) for _ in range(self.n)]
        # 自己到自己信息素为 0
        for i in range(self.n):
            self.P[i][i] = 0

        self.int_data = int_data
        self.int_data["costs"] = []

    def _select_next_city(self, current_city: int, visited_cities: list):
        """
        让一只蚂蚁选择下一个城市

        :param current_city: 当前城市
        :param visited_cities: 已经访问过的城市
        :return: 下一个城市
        """
        pheromone = self.P[current_city]  # 从当前城市到其他城市的信息素
        distance = self.G[current_city]  # 从当前城市到其他城市的距离

        # 计算当前城市到每个城市的概率

        # 分子，每个城市的权重
        city_weights = [
            (
                # 前者是信息素浓度的重要性
                # 后者是启发式信息，通常是路径长度的导数，为了防止除以 0，倒数分母加上 1e-10
                (pheromone[i] ** ALPHA) * ((1.0 / (distance[i] + 1e-10)) ** BETA)
                # 已访问的城市的选择概率直接归 0
                if not visited_cities[i]
                else 0
            )
            for i in range(self.n)
        ]
        # 计算到每个城市的概率
        city_weights_sum = sum(city_weights)
        city_prob = [weight / city_weights_sum for weight in city_weights]

        # 选择下一个城市
        next_city = random.choices(range(self.n), weights=city_prob, k=1)[0]

        return next_city

    def _path_cost(self, path: list) -> int:
        """
        计算路径 path 的总成本

        :param path: 路径
        """
        cost = 0
        for i in range(len(path)):
            cost += self.G[path[i]][path[(i + 1) % len(path)]]
        return cost

    def _update_pheromone(self, ants_paths: list, ants_lengths: list):
        """
        更新信息素

        :param ants_paths: 每只蚂蚁走过的路径
        :param ants_lengths: 每只蚂蚁走过的路径长度
        """

        # Step.1 信息素挥发
        for i in range(self.n):
            for j in range(self.n):
                # 挥发后只剩 1-p 的信息素
                self.P[i][j] *= 1 - EVAPORATION_RATE

        # Step.2 计算每条路径上，所有蚂蚁累积的信息素，作为增量
        for path, length in zip(ants_paths, ants_lengths):
            for i in range(len(path)):
                # 这只蚂蚁在城市 i 和 i+1 的路径上留下的信息素
                delta_pheromone = Q / length
                self.P[path[i]][path[(i + 1) % len(path)]] += delta_pheromone
                # 因为是无向图，所以要双向增加信息素
                self.P[path[(i + 1) % len(path)]][path[i]] += delta_pheromone

    def solve(self):
        """
        蚁群算法

        :return: (最优路径 list，最优成本 int)
        """
        # 结果
        best_path = []
        min_cost = INF_DIST



        for _ in range(NUM_ITERATIONS):
            ants_paths = []
            ants_lengths = []

            for _ in range(NUM_ANTS):
                # 随机初始化每只蚂蚁的路径

                # 每只蚂蚁访问过的城市列表
                visited_cities = [False] * self.n
                # 随机从一个城市出发
                start_city = random.randint(0, self.n - 1)
                visited_cities[start_city] = True
                path = [start_city]

                # 让这个蚂蚁想办法走一圈
                for _ in range(self.n - 1):
                    # 目前所在的城市
                    current_city = path[-1]
                    # 选择下一个城市
                    next_city = self._select_next_city(current_city, visited_cities)
                    visited_cities[next_city] = True
                    path.append(next_city)

                # 计算路径长度
                path_length = self._path_cost(path)

                # 存储路径和路径长度
                ants_paths.append(path)
                ants_lengths.append(path_length)

            # 每次迭代后更新信息素
            self._update_pheromone(ants_paths, ants_lengths)

            # 找到当前迭代的最优路径
            current_min_len = min(ants_lengths)

            if self.int_data is None:
                self.int_data["costs"] = [current_min_len]
            else:
                self.int_data["costs"].append(current_min_len)

            if current_min_len < min_cost:
                ind = ants_lengths.index(current_min_len)
                best_path = ants_paths[ind]
                min_cost = current_min_len

        return best_path, min_cost
