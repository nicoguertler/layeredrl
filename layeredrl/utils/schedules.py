from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class Schedule(ABC):
    """Base class for (learning rate) schedules."""

    @abstractmethod
    def __call__(self, t: int, lr: float = None) -> float:
        pass


class RampSchedule(Schedule):
    def __init__(self, start: float, end: float, warm_up: int, duration: int):
        self.warm_up = warm_up
        self.start = start
        self.end = end
        self.duration = duration

    def __call__(self, t: int, lr: float = None) -> float:
        if t < self.warm_up:
            return self.start
        else:
            return self.start + (self.end - self.start) * min(
                (t - self.warm_up) / self.duration, 1.0
            )


class ZeroThenLinearSchedule(Schedule):
    def __init__(
        self, zero_duration: int, duration: int, start_value: float, end_value: float
    ):
        self.zero_duration = zero_duration
        self.duration = duration
        self.start_value = start_value
        self.end_value = end_value

    def __call__(self, t: int, lr: float = None) -> float:
        if t < self.zero_duration:
            return 0.0
        else:
            return self.start_value + (self.end_value - self.start_value) * min(
                (t - self.zero_duration) / self.duration, 1.0
            )


class PiecewiseLinearSchedule(Schedule):
    def __init__(self, points: List[Tuple[int, float]]):
        self.points = points

    def __call__(self, t: int, lr: float = None) -> float:
        for i in range(len(self.points) - 1):
            if self.points[i][0] <= t < self.points[i + 1][0]:
                return self.points[i][1] + (
                    self.points[i + 1][1] - self.points[i][1]
                ) * (
                    (t - self.points[i][0])
                    / (self.points[i + 1][0] - self.points[i][0])
                )
        return self.points[-1][1]


class PiecewiseLogLinearSchedule(Schedule):
    def __init__(self, points: List[Tuple[int, float]]):
        self.points = points

    def __call__(self, t: int, lr: float = None) -> float:
        for i in range(len(self.points) - 1):
            if self.points[i][0] <= t < self.points[i + 1][0]:
                if self.points[i][1] == 0 or self.points[i + 1][1] == 0:
                    return 0
                log_lr_1 = np.log(self.points[i][1])
                log_lr_2 = np.log(self.points[i + 1][1])
                return np.exp(
                    log_lr_1
                    + (log_lr_2 - log_lr_1)
                    * (
                        (t - self.points[i][0])
                        / (self.points[i + 1][0] - self.points[i][0])
                    )
                )
        return self.points[-1][1]


class ConstantSchedule(Schedule):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, t: int, lr: float = None) -> float:
        return self.value
