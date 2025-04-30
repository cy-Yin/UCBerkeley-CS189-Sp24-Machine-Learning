from .utils import LayerTest
from neural_networks.activations import Linear, Sigmoid, TanH, ReLU, SoftMax, ArcTan

class TestLinear(LayerTest):

    LayerCls  = Linear

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")


class TestSigmoid(LayerTest):

    LayerCls  = Sigmoid

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")


class TestTanH(LayerTest):

    LayerCls  = TanH
    BatchSizes = (64,)
    InputSizes = ((31,),)

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")


class TestReLU(LayerTest):

    LayerCls  = ReLU

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")


class TestSoftMax(LayerTest):

    LayerCls  = SoftMax

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")


class TestArcTan(LayerTest):

    LayerCls  = ArcTan
    BatchSizes = (64,)
    InputSizes = ((31,),)

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")

