from .utils import LayerTest
from neural_networks.losses import CrossEntropy

class TestCrossEntropy(LayerTest):

    LayerCls  = CrossEntropy
    LayerConfigs = ({"name": "cross_entropy"},)

    def test_forward(self):
        return self._test(mode="forward")

    def test_backward(self):
        return self._test(mode="backward")