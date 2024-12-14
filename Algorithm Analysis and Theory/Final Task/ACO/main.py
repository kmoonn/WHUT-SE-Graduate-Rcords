import random
import numpy as np
import matplotlib.pyplot as plt
import itertools


# 计算距离矩阵
def calculate_distance_matrix(points):
    num_points = len(points)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i][j] = np.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
    return distance_matrix


# 蚁群算法参数
class ACO:
    def __init__(self, distance_matrix, num_ants=10, alpha=1.0, beta=2.0, rho=0.5, Q=100, max_iterations=5):
        self.distance_matrix = distance_matrix  # 城市的距离矩阵
        self.num_ants = num_ants  # 蚂蚁数量
        self.alpha = alpha  # 信息素重要性因子
        self.beta = beta  # 启发式信息重要性因子
        self.rho = rho  # 信息素挥发率
        self.Q = Q  # 信息素增加量
        self.max_iterations = max_iterations  # 最大迭代次数
        self.num_cities = len(distance_matrix)  # 城市数量

        # 初始化信息素矩阵
        self.pheromone_matrix = np.ones((self.num_cities, self.num_cities))

    def run(self):
        best_path = None
        best_length = float('inf')
        path_length_record = []

        # 多次迭代
        for iteration in range(self.max_iterations):
            all_paths = []  # 存储所有蚂蚁的路径
            all_lengths = []  # 存储所有蚂蚁的路径长度

            # 每只蚂蚁构造路径
            for ant in range(self.num_ants):
                path = self.construct_path()  # 构造路径
                length = self.calculate_path_length(path)  # 计算路径长度
                all_paths.append(path)  # 保存路径
                all_lengths.append(length)  # 保存路径长度

                # 更新最优解
                if length < best_length:
                    best_length = length
                    best_path = path
                    path_length_record.append(best_length)

            # 更新信息素
            self.update_pheromones(all_paths, all_lengths)

        return best_path, best_length, path_length_record

    def construct_path(self):
        path = [0]  # 从配送中心开始，索引为0
        visited = set(path)  # 只需要记录已访问的城市

        # 逐步构造路径
        for _ in range(1, self.num_cities):  # 仅遍历配送点
            current_city = path[-1]  # 当前城市是路径中的最后一个城市
            next_city = self.select_next_city(current_city, visited)  # 选择下一个城市
            path.append(next_city)  # 添加到路径中
            visited.add(next_city)  # 标记为已访问

        return path  # 返回构造的路径

    def select_next_city(self, current_city, visited):
        probabilities = []  # 存储选择下一个城市的概率
        for city in range(self.num_cities):
            if city not in visited:  # 如果城市未被访问
                pheromone = self.pheromone_matrix[current_city][city]  # 获取当前城市到目标城市的信息素浓度
                distance = self.distance_matrix[current_city][city]  # 获取当前城市到目标城市的距离
                # 根据信息素和距离计算选择概率
                probabilities.append((pheromone ** self.alpha) * ((1.0 / distance) ** self.beta))
            else:
                probabilities.append(0)  # 已访问城市的概率为0

        total_prob = sum(probabilities)  # 计算总概率
        probabilities = [p / total_prob for p in probabilities]  # 归一化概率

        return np.random.choice(range(self.num_cities), p=probabilities)  # 根据概率选择下一个城市

    def calculate_path_length(self, path):
        length = 0.0  # 初始化路径长度
        for i in range(len(path) - 1):
            length += self.distance_matrix[path[i]][path[i + 1]]  # 累加两城市之间的距离
        # 将最后一个城市和配送中心相连以形成闭合路径
        length += self.distance_matrix[path[-1]][0]  # 从最后一个城市回到配送中心
        return length  # 返回总路径长度

    def update_pheromones(self, all_paths, all_lengths):
        # 信息素挥发
        self.pheromone_matrix *= (1 - self.rho)
        # 更新信息素
        for path, length in zip(all_paths, all_lengths):
            # 信息素增加量与路径长度成反比
            pheromone_increase = self.Q / length  # 根据路径长度计算信息素增加量
            for i in range(len(path) - 1):
                self.pheromone_matrix[path[i]][path[i + 1]] += pheromone_increase  # 更新城市之间的信息素
            # 闭合路径：更新最后一个城市和配送中心之间的信息素
            self.pheromone_matrix[path[-1]][0] += pheromone_increase


# 主程序
if __name__ == "__main__":
    # 设置城市数量和随机生成城市坐标
    random.seed(24)
    num_cities = 50
    points = [(random.randint(20, 99), random.randint(20, 99)) for _ in range(num_cities)]
    # 添加配送中心坐标
    distribution_center = (60, 60)
    points.insert(0, distribution_center)  # 在列表开头插入配送中心

    # 获取距离矩阵
    distance_matrix = calculate_distance_matrix(points)

    # 使用网格搜索方式进行调参
    # 定义参数范围
    num_ants = [10, 20, 30]
    alpha = [1.0, 1.5]
    beta = [2.0, 3.0]
    rho = [0.4, 0.5]
    max_iterations = [100, 200, 300, 400]
    # 生成参数组合
    param_grid = itertools.product(num_ants, alpha, beta, rho, max_iterations)
    # 遍历每个参数组合
    best_length = float('inf')
    best_path = []
    best_params = None
    best_Optimizion_record = []
    for params in param_grid:
        print('==')

        num_ants, alpha, beta, rho, max_iterations = params
        aco = ACO(distance_matrix, num_ants=num_ants, alpha=alpha, beta=beta, rho=rho, Q=100,
                  max_iterations=max_iterations)
        current_path, current_length, path_length_record = aco.run()
        if current_length < best_length:
            best_length = current_length
            best_path = current_path
            best_params = params
            best_Optimization_record = path_length_record

    print("Best Path:", best_path)
    print("Best Length:", best_length)
    print("Best Params:", best_params)

    # 可视化路径
    best_points = [points[i] for i in best_path]
    best_points.append(best_points[0])  # 返回起点
    plt.plot(*zip(*best_points), marker='o', label='Best Path')
    plt.scatter(*zip(*points[1:]), color='blue', label='Delivery Points')
    plt.scatter(*distribution_center, color='red', s=300, marker='*', label='Distribution Center')  # 标记配送中心
    plt.title('Best Delivery Path Found by ACO')
    plt.legend()
    plt.show()

    # 可视化优化过程
    plt.plot(range(len(best_Optimization_record)), best_Optimization_record)
    plt.text(0, best_Optimization_record[0], f'{round(best_Optimization_record[0], 3)}', fontsize=10,
             verticalalignment='top')
    plt.text(len(best_Optimization_record) - 1, best_Optimization_record[-1],
             f'{round(best_Optimization_record[-1], 3)}', fontsize=10, verticalalignment='top')
    plt.title('Optimization Process')
    plt.xlabel('Iterations')
    plt.ylabel('Path Length')
    plt.show()
