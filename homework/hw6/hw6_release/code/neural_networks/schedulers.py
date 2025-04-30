"""
Author: Sophia Sanborn
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas
"""

import numpy as np
from abc import ABC, abstractmethod
import math


def initialize_scheduler(name, lr, decay=None, stage_length=None, staircase=None):
    if name == "constant":
        return Constant(lr=lr)
    elif name == "exponential":
        return Exponential(
            lr=lr, decay=decay, stage_length=stage_length, staircase=None
        )
    else:
        raise NotImplementedError("{} scheduler is not implemented".format(name))


class Scheduler(ABC):
    def __call__(self, epoch):
        return self.scheduled_lr(epoch)

    @abstractmethod
    def scheduled_lr(self, epoch=None):
        pass


class Constant(Scheduler):
    def __init__(self, lr=0.01):
        self.lr = lr

    def scheduled_lr(self, epoch):
        return self.lr


class Exponential(Scheduler):
    def __init__(self, lr=0.01, decay=0.9, stage_length=1000, staircase=False):
        self.lr = lr
        self.decay = decay
        self.stage_length = stage_length
        self.staircase = staircase

    def scheduled_lr(self, epoch):
        if self.staircase:
            stage = math.floor(epoch / self.stage_length)
        else:
            stage = epoch / self.stage_length

        return self.lr * self.decay ** stage
