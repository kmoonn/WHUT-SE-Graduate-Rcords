from pprint import pprint

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 生成经典小世界网络
def generate_small_world(n, k, p):
    """
    生成小世界网络
    :param n: 节点数
    :param k: 每个节点的平均连接数
    :param p: 重连概率
    :return: 小世界网络图
    """
    return nx.watts_strogatz_graph(n, k, p)

# 改进模型（动态增长模拟）
def generate_dynamic_small_world(n, initial_nodes, k, p, steps):
    """
    动态增长的小世界模型
    :param n: 总节点数
    :param initial_nodes: 初始节点数
    :param k: 平均连接数
    :param p: 重连概率
    :param steps: 动态增长的时间步数
    :return: 动态演化后的网络
    """
    graph = nx.watts_strogatz_graph(initial_nodes, k, p)
    for _ in range(steps):
        new_node = len(graph)
        graph.add_node(new_node)
        # 按优先连接规则添加边
        targets = np.random.choice(graph.nodes(), k, replace=False)
        for target in targets:
            graph.add_edge(new_node, target)
    return graph


# 网络分析函数
def analyze_network(graph):
    """
    分析网络的基本特性
    :param graph: 网络图
    :return: 指标字典
    """
    avg_clustering = nx.average_clustering(graph)
    avg_path_length = nx.average_shortest_path_length(graph)
    degree_dist = [deg for _, deg in graph.degree()]
    return {
        "平均聚类系数": avg_clustering,
        "平均路径长度": avg_path_length,
        "度分布": degree_dist
    }

# 可视化网络
def plot_network(graph, title="Network Visualization"):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=False, node_size=20)
    plt.title(title)
    plt.show()

# 度分布可视化
def plot_degree_distribution(degree_dist):
    plt.figure(figsize=(8, 6))
    plt.hist(degree_dist, bins=30, color='blue', alpha=0.7)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == '__main__':
    # 实验参数
    N = 20  # 总节点数
    K = 4    # 平均连接数
    P = 0.1  # 重连概率
    INITIAL_NODES = 50  # 初始节点数
    STEPS = 450  # 动态增长步数

    # 生成网络
    classic_graph = generate_small_world(N, K, P)
    dynamic_graph = generate_dynamic_small_world(N, INITIAL_NODES, K, P, STEPS)

    # 分析网络
    classic_analysis = analyze_network(classic_graph)
    dynamic_analysis = analyze_network(dynamic_graph)

    # 输出分析结果
    print("经典小世界模型:")
    pprint(classic_analysis)
    print("\n改进小世界模型:")
    pprint(dynamic_analysis)

    # 可视化
    plot_network(classic_graph, "Classic Small-World Network")
    plot_network(dynamic_graph, "Dynamic Small-World Network")
    plot_degree_distribution(dynamic_analysis["度分布"])