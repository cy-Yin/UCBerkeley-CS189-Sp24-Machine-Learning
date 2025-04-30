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


def initialize_weights(name, activation=None, mode="fan_in"):
    if name == "zeros":
        return Zeros()
    elif name == "ones":
        return Ones()
    elif name == "identity":
        return Identity()
    elif name == "uniform":
        return Uniform()
    elif name == "normal":
        return Normal()
    elif name == "constant":
        return Constant()
    elif name == "sparse":
        return Sparse()
    elif name == "he_uniform":
        return HeUniform(activation=activation, mode=mode)
    elif name == "he_normal":
        return HeNormal(activation=activation, mode=mode)
    elif name == "xavier_uniform":
        return XavierUniform(activation=activation)
    elif name == "xavier_normal":
        return XavierNormal(activation=activation)
    else:
        raise NotImplementedError


def _calculate_gain(activation, param=None):
    """
    Adapted from https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain
    """
    linear_fns = [
        "linear",
        "conv2d",
    ]
    if (
        activation in linear_fns
        or activation == "sigmoid"
        or activation == "softmax"
    ):
        return 1.0
    elif activation == "tanh":
        return 5.0 / 3.0
    elif activation == "relu":
        return math.sqrt(2.0)
    else:
        return 1.0


def _get_fan(shape, mode="sum"):
    fan_in, fan_out = shape[0], shape[-1]
    if mode == "fan_in":
        return fan_in
    elif mode == "fan_out":
        return fan_out
    elif mode == "sum":
        return fan_in + fan_out
    elif mode == "separate":
        return fan_in, fan_out
    else:
        raise ValueError("Mode must be one of fan_in, fan_out, sum, or separate")


class WeightInitializer(ABC):
    @abstractmethod
    def __call__(self):
        pass


class Zeros(WeightInitializer):
    def __call__(self, shape):
        W = np.zeros(shape=shape)
        return W


class Ones(WeightInitializer):
    def __call__(self, shape):
        W = np.ones(shape=shape)
        return W


class Identity(WeightInitializer):
    def __call__(self, shape):
        fan_in, fan_out = _get_fan(shape, mode="separate")
        if fan_in != fan_out:
            raise ValueError(
                "Weight matrix shape must be square for identity initialization"
            )
        W = np.identity(n=fan_in)
        return W


class Uniform(WeightInitializer):
    def __init__(self, low=-1.0, high=1.0):
        self.low = low
        self.high = high

    def __call__(self, shape):
        W = np.random.uniform(self.low, self.high, size=shape)
        return W


class Normal(WeightInitializer):
    def __init__(self, mean=0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, shape):
        W = np.random.normal(self.mean, self.std, size=shape)
        return W


class Constant(WeightInitializer):
    def __init__(self, val=0.5):
        self.val = val

    def __call__(self, shape):
        W = np.full(shape, self.val)
        return W


class Preset(WeightInitializer):
    def __call__(self, preset_matrix):
        return preset_matrix


class Sparse(WeightInitializer):
    def __init__(self, sparsity=0.1, std=0.01):
        self.sparsity = sparsity
        self.std = std

    def __call__(self, shape):
        n_rows, n_cols = shape
        n_zeros = int(math.ceil(n_rows * self.sparsity))

        W = np.random.normal(0, self.std, size=shape)
        for col_idx in range(n_cols):
            row_idx = np.arange(n_rows)
            np.random.shuffle(row_idx)
            zero_idx = row_idx[:n_zeros]
            W[zero_idx, col_idx] = 0
        return W


class XavierUniform(WeightInitializer):
    def __init__(self, activation=None):
        self.activation = activation

    def __call__(self, shape):
        fan = _get_fan(shape, mode="sum")
        gain = _calculate_gain(self.activation)
        std = gain * math.sqrt(2.0 / (fan))
        a = math.sqrt(3.0) * std
        W = np.random.uniform(-a, a, size=shape)
        return W


class XavierNormal(WeightInitializer):
    def __init__(self, activation=None):
        self.activation = activation

    def __call__(self, shape):
        fan = _get_fan(shape, mode="sum")
        gain = _calculate_gain(self.activation)
        std = gain * math.sqrt(2.0 / (fan))
        W = np.random.normal(0, std, size=shape)
        return W


class HeUniform(WeightInitializer):
    def __init__(self, activation=None, mode="fan_in"):
        self.activation = activation
        self.mode = mode

    def __call__(self, shape):
        fan = _get_fan(shape, mode=self.mode)
        gain = _calculate_gain(self.activation)
        std = gain / math.sqrt(fan)
        a = math.sqrt(3.0) * std
        W = np.random.uniform(-a, a, size=shape)
        return W


class HeNormal(WeightInitializer):
    def __init__(self, activation=None, mode="fan_in"):
        self.activation = activation
        self.mode = mode

    def __call__(self, shape):
        fan = _get_fan(shape, mode=self.mode)
        gain = _calculate_gain(self.activation)
        std = gain / math.sqrt(fan)
        W = np.random.normal(0, std, size=shape)
        return W
