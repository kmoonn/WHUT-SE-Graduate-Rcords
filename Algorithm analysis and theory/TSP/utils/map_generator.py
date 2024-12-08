# _*_ coding : utf-8 _*_
# @Time : 2024/11/6 下午8:03
# @Author : Kmoon_Hs
# @File : map_generator

import matplotlib.pyplot as plt
import networkx as nx
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 定义武汉旅游景点及其经纬度
wuhan_scenic_spots = {
    "武汉植物园": (30.537, 114.436),
    "东湖绿道": (30.558, 114.405),
    "欢乐谷": (30.586, 114.425),
    "湖北省博物馆": (30.563, 114.385),
    "武汉大学": (30.536, 114.360),
    "楚河汉街": (30.566, 114.344),
    "黄鹤楼": (30.547, 114.309),
    "武汉长江大桥": (30.562, 114.279),
    "汉口江滩": (30.587, 114.294),
    "归元寺": (30.545, 114.258)
}


# 生成无向带权图，控制每个节点的出度
def generate_weighted_graph_with_edges(spots, problem):
    G = nx.Graph()  # 创建一个无向图
    names = list(spots.keys())

    edges = problem['edges']

    # 将景点作为节点加入图中
    for name in names:
        G.add_node(name, pos=spots[name])

    # 生成边和权重
    for item in edges:
        node1, node2, weight = item
        G.add_edge(names[node1], names[node2], weight=weight)

    return G


def generate_possible_path(spots, result):
    names = list(spots.keys())

    G_directed = nx.DiGraph()
    possible_path = result[0]
    # 定义所有边的默认颜色
    edge_colors = ["black" for _ in G_directed.edges()]

    # 定义需要更改颜色的边
    special_edges = []
    for i in range(len(possible_path) - 1):
        special_edges.append((names[possible_path[i]], names[possible_path[i + 1]]))

    # 给特定边设置颜色
    for i, edge in enumerate(G_directed.edges()):
        if edge in special_edges or (edge[1], edge[0]) in special_edges:
            edge_colors[i] = "red"  # 将特定边设置为红色
    G_directed.add_edges_from(special_edges)

    return G_directed


# 可视化图
def plot_graph(G, G_directed, spots, problem, algo):
    names = list(spots.keys())
    pos = nx.get_node_attributes(G, 'pos')  # 获取节点的坐标
    weights = nx.get_edge_attributes(G, 'weight')  # 获取边的权重

    # 创建一个多重图来组合无向图和有向图
    G_mixed = nx.MultiDiGraph()
    G_mixed.add_edges_from(G.edges())  # 添加无向边
    G_mixed.add_edges_from(G_directed.edges())  # 添加有向边

    # 定义所有节点的默认颜色
    node_colors = ["skyblue" for _ in G_mixed.nodes()]

    # 定义需要更改颜色的节点
    special_nodes = names[problem['possible_path'][0]]

    # 给特定节点设置颜色
    for i, node in enumerate(G_mixed.nodes()):
        if node in special_nodes:
            node_colors[i] = "orange"  # 将特定节点设置为橙色

    # 绘制无向边
    nx.draw_networkx_edges(G_mixed, pos, edgelist=G.edges(), edge_color="black", arrows=False)
    # 绘制有向边
    nx.draw_networkx_edges(G_mixed, pos, edgelist=G_directed.edges(), edge_color="red", arrows=True)
    # 绘制节点
    nx.draw_networkx_nodes(G_mixed, pos, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_labels(G_mixed, pos)
    # 显示边的权重
    nx.draw_networkx_edge_labels(G_mixed, pos, edge_labels=weights, font_size=8)

    plt.title(f'Result of {algo}')
    plt.show()


def show(problem, result, algo):
    # 读取 TSP 问题
    print(problem)

    # 生成图
    G = generate_weighted_graph_with_edges(wuhan_scenic_spots, problem)
    G_directed = generate_possible_path(wuhan_scenic_spots, result)

    # 可视化图
    plot_graph(G, G_directed, wuhan_scenic_spots, problem, algo)
