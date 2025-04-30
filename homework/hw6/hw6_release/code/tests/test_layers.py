import random
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from .utils import LayerTest

from neural_networks.layers import Conv2D, Flatten, FullyConnected, Pool2D


class TestFullyConnected(LayerTest):

    LayerCls  = FullyConnected
    LayerConfigs = (
        {"n_out": 128, "activation": "relu"},
        {"n_out": 37, "activation": "tanh"},
        {"n_out": 256, "activation": "tanh"},
    )

    def test_init_params(self):
        for _ in range(50):
            foo = random.randint(1, 1000)
            bar = random.randint(1, 1000)
            baz = (-1, bar)
            fc_layer = FullyConnected(foo, "linear")
            fc_layer._init_parameters(baz)

            self.assertEqual(
                fc_layer.parameters["W"].shape,
                (bar, foo),
                "Incorrect shape of parameter `W`.",
            )
            self.assertEqual(
                fc_layer.parameters["b"].shape,
                (1, foo),
                "Incorrect shape of parameter `b`.",
            )
            assert_almost_equal(
                fc_layer.gradients["W"],
                0,
                err_msg="Incorrect initialization of gradient wrt `W`.",
            )
            self.assertEqual(
                fc_layer.gradients["W"].shape,
                (bar, foo),
                "Incorrect shape of gradient wrt `W`.",
            )
            assert_almost_equal(
                fc_layer.gradients["b"],
                0,
                err_msg="Incorrect initialization of gradient wrt `b`.",
            )
            self.assertEqual(
                fc_layer.gradients["b"].shape,
                (1, foo),
                "Incorrect shape of gradient wrt `b`.",
            )

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")


class TestConv2D(LayerTest):

    LayerCls = Conv2D
    LayerConfigs = (
        {"n_out": 32, "activation": "relu", "kernel_shape": (3, 3)},
        {"n_out": 32, "activation": "relu", "kernel_shape": (3, 5)},
        {"n_out": 7, "activation": "relu", "kernel_shape": (5, 5)},
        {"n_out": 7, "activation": "relu", "kernel_shape": (5, 5), "stride": 2},
        {"n_out": 18, "activation": "tanh", "kernel_shape": (3, 3), "stride": 3},
    )
    InputSizes = ((16, 16, 3), (32,24,21),)
    BatchSizes = (16,)

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")


class TestPool2D(LayerTest):

    LayerCls = Pool2D
    LayerConfigs = (
        {"kernel_shape": (3, 3), "stride": 2},
        {"kernel_shape": (2, 2), "stride": 2},
        {"kernel_shape": (2, 2), "mode": "average"},
        {"kernel_shape": (3, 5), "stride": 4, "pad": "valid"},
        {"kernel_shape": (5, 5), "mode": "average", "pad": 2},
    )
    InputSizes = ((12, 12, 8), (44,48,4),)
    BatchSizes = (24,)

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")


class TestFlatten(LayerTest):

    LayerCls = Flatten
    LayerConfigs = (
        {},
    )
    InputSizes = ((16,13,5), )
    BatchSizes = (18,)

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
       return self._test(mode="backward")
