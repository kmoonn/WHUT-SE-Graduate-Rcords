"""
Solution 基类
"""


class SolutionBase:
    # 算法标识名
    algorithm_id = "base"
    algorithm_name = "Base"

    def __init__(self):
        pass

    def solve(self):
        """
        执行主算法

        :return: (最优路径 list，最优成本 int, 中间数据 { <br>
                    "keys":[(数据 key，数据名), ...],  <br>
                    "int_data": { 数据key: 数据 list} <br>
            })
        """
        pass
