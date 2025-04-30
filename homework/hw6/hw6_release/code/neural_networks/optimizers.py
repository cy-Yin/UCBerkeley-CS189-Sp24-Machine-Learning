"""
Author: Sophia Sanborn
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas
"""

import numpy as np
from abc import ABC, abstractmethod
from neural_networks.schedulers import initialize_scheduler


def initialize_optimizer(
    name,
    lr,
    lr_scheduler=None,
    momentum=None,
    clip_norm=None,
    lr_decay=None,
    staircase=None,
    stage_length=None,
):
    if name == "SGD":
        return SGD(
            lr=lr,
            lr_scheduler=lr_scheduler,
            momentum=momentum,
            clip_norm=clip_norm,
            lr_decay=lr_decay,
            staircase=staircase,
            stage_length=stage_length,
        )
    else:
        raise NotImplementedError


class Optimizer(ABC):
    def __init__(self):
        self.lr = None
        self.lr_scheduler = None


class SGD(Optimizer):
    def __init__(
        self,
        lr,
        lr_scheduler,
        momentum=0.0,
        clip_norm=None,
        lr_decay=0.9,
        stage_length=None,
        staircase=None,
    ):
        self.lr = lr
        self.lr_scheduler = initialize_scheduler(
            lr_scheduler,
            lr=lr,
            decay=lr_decay,
            stage_length=stage_length,
            staircase=staircase,
        )
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.cache = {}

    def update(self, param_name, param, param_grad, epoch):
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)

        if self.clip_norm is not None:
            if np.linalg.norm(param_grad) > self.clip_norm:
                param_grad = (
                    param_grad * self.clip_norm / np.linalg.norm(param_grad)
                )

        lr = self.lr_scheduler(epoch)
        delta = (
            self.momentum * self.cache[param_name]
            + lr * param_grad
        )
        self.cache[param_name] = delta
        return delta
