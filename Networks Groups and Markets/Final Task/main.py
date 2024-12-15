import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# 小世界模型构建函数
def generate_ws_network(N, k, p):
    """
    生成小世界网络
    :param N: 节点数量
    :param k: 每个节点的邻居数量
    :param p: 边随机重连的概率
    :return: 小世界网络图
    """
    return nx.watts_strogatz_graph(N, k, p)

# 模拟数据采集网络
def load_web_network(filepath):
    """
    加载万维网采集数据并转换为网络图
    :param filepath: 采集数据的邻接矩阵文件路径
    :return: 万维网网络图
    """
    adj_matrix = np.loadtxt(filepath, delimiter=',')
    return nx.from_numpy_array(adj_matrix)

# 网络特性分析函数
def analyze_network(graph):
    """
    分析网络的关键特性
    :param graph: 输入的网络图
    :return: 平均路径长度、聚类系数、度分布
    """
    avg_path_length = nx.average_shortest_path_length(graph)
    clustering_coefficient = nx.average_clustering(graph)
    degree_sequence = [degree for _, degree in graph.degree()]
    return avg_path_length, clustering_coefficient, degree_sequence

# 模拟实验
N = 1000
k = 6
p_values = [0.01, 0.05, 0.1]
results = []

for p in p_values:
    ws_network = generate_ws_network(N, k, p)
    avg_path, cluster_coeff, degrees = analyze_network(ws_network)
    results.append((p, avg_path, cluster_coeff))

# 结果可视化
plt.figure(figsize=(10, 6))
for i, p in enumerate(p_values):
    degrees = [degree for _, degree in generate_ws_network(N, k, p).degree()]
    degree_count = Counter(degrees)
    plt.bar(degree_count.keys(), degree_count.values(), alpha=0.5, label=f'p={p}')

plt.title('Degree Distribution for Small-World Networks')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 加载万维网采集数据并分析
web_network = load_web_network("web_network.csv")
web_avg_path, web_cluster_coeff, web_degrees = analyze_network(web_network)

# 输出特性
print(f"采集数据的平均路径长度: {web_avg_path:.4f}")
print(f"采集数据的平均聚类系数: {web_cluster_coeff:.4f}")
