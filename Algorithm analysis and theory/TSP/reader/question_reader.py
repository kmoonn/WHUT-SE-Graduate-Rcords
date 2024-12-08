"""
用于从 md 文件中按行读取问题输入 
"""

import re
from random import shuffle

INPUT_REGEX = re.compile(r"```([\S\s]+?)```")


def read_question(file_path) -> dict:
    """
    读取 md 文件 file_path 中的问题输入

    :param file_path: md 文件路径
    :return: 问题输入，一个字典：{n: 景点数量，m: 边数量，edges: [(u, v, weight), (u, v, weight), ...], best_cost: 最优成本, possible_path: 可能的最优路径}
    """
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            file_content = f.read()
            input_matcher = INPUT_REGEX.finditer(file_content)
            try:
                input_str = next(input_matcher).group(1).strip()
            except StopIteration:
                raise Exception(f"未找到问题输入")
            input_lines = input_str.splitlines()
            # 解析成问题输入
            question = {}
            # 第一行为两个整数，分别为景点数量和边数量
            first_line = input_lines[0].split()
            if len(first_line) != 2:
                raise Exception(f"第一行格式错误：{first_line}")
            question["n"] = int(first_line[0])
            question["m"] = int(first_line[1])
            question["edges"] = []
            # 从第二行开始为问题输入
            for line in input_lines[1:]:
                question["edges"].append(tuple(map(int, line.split())))
            # 打乱边的输入
            shuffle(question["edges"])

            try:
                best_solution_str = next(input_matcher).group(1).strip()
            except StopIteration:
                raise Exception(f"未找到文档中的最优解")
            best_solution_lines = best_solution_str.splitlines()
            question["best_cost"] = int(best_solution_lines[0].strip())
            question["possible_path"] = list(
                map(int, best_solution_lines[1].strip().split())
            )
            return question

    except Exception as e:
        print(f"读取 {file_path} 失败：{e}")


def read_questions(file_paths) -> list[dict]:
    """
    读取多个 md 文件中的问题输入

    :param file_paths: md 文件路径列表
    :return: 问题输入，一个数组：[{n: 景点数量，m: 边数量，edges: [(u, v, weight), (u, v, weight), ...], best_cost: 最优成本, possible_path: 可能的最优路径}, ...]
    """
    return [read_question(file_path) for file_path in file_paths]
