import os
import unittest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import numpy as np
import itertools

from neural_networks.layers import Layer
from neural_networks.activations import Activation
from neural_networks.losses import CrossEntropy, Loss

class LayerTest(unittest.TestCase):

    LayerCls = None
    InputRange = (-1, 1)

    LayerConfigs = ({},)
    BatchSizes = (72, 128, 223)
    InputSizes = ((61,), (311,),)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls_name = self.__class__.__name__
        data_path = os.path.join(os.path.dirname(__file__), 'data', cls_name + '.npz')
        self.test_configs = list(itertools.product(self.LayerConfigs, self.BatchSizes, self.InputSizes))
        if os.path.exists(data_path):
            pass # everything ok
        else:
            raise ValueError("Cannot find testing data.")

        self.test_data = np.load(data_path)

    def _test(self, mode="forward"):
        """
        mode is one of forward or backward
        """
        for i, (layer_config, _, _) in enumerate(self.test_configs):
            layer = self.LayerCls(**layer_config)
            input_data = self.test_data[str(i) + "input"]
            output_data = self.test_data[str(i) + "output"]
            upstream_grad = self.test_data[str(i) + "upstream_grad"]
            backward_data = self.test_data[str(i) + "backward"]
            if hasattr(layer, "parameters"):
                layer.forward(*input_data)
                if "W" in layer.parameters:
                    layer.parameters["W"] = self.test_data[str(i) + "W"]
                if "b" in layer.parameters:
                    layer.parameters["b"] = self.test_data[str(i) + "b"]    
            layer_output = layer.forward(*input_data)
            if mode == "forward":
                assert_almost_equal(output_data, layer_output, decimal=4)
            elif mode == "backward":
                if isinstance(layer, Activation):
                    backward_output = layer.backward(*input_data, dY=upstream_grad)
                elif isinstance(layer, Loss):
                    backward_output = layer.backward(*input_data)
                else:
                    backward_output = layer.backward(upstream_grad)                
                assert_almost_equal(backward_data, backward_output, decimal=4)
        return True