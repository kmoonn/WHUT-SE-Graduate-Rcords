"""
简单的程序运行计时器
"""

from time import perf_counter_ns


class Timer:
    def __init__(self):
        self.start_time = None
        self.time_diff = 0

    def start(self):
        """
        开始计时
        """
        self.start_time = perf_counter_ns()

    def stop(self):
        """
        停止计时
        """
        self.time_diff = perf_counter_ns() - self.start_time

    def time_consumed_ms(self):
        """
        上一个计时区间的耗时（毫秒）
        """
        return self.time_diff / 1e6
