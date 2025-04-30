
"""
Step 1: Define layer arguments

- Define the arguments for each layer in an attribute dictionary (AttrDict).
- An attribute dictionary is exactly like a dictionary, except you can access the values as attributes rather than keys...for cleaner code :)
- See layers.py for the arguments expected by each layer type.
"""

from neural_networks.utils import AttrDict

conv1 = AttrDict(
    {
        "name": "conv2d",
        "n_out": 6,
        "kernel_shape": (5, 5),
        "stride": 1,
        "pad": "same",
        "activation": "relu",
        "weight_init": "he_uniform",
    }
)

pool1 = AttrDict(
    {
        "name": "pool2d",
        "kernel_shape": (2, 2),
        "mode": "max",
        "stride": 2,
        "pad": "valid"
    }
)

conv2 = AttrDict(
    {
        "name": "conv2d",
        "n_out": 16,
        "kernel_shape": (5, 5),
        "stride": 1,
        "pad": "valid",
        "activation": "relu",
        "weight_init": "he_uniform",
    }
)

pool2= AttrDict(
    {
        "name": "pool2d",
        "kernel_shape": (2, 2),
        "mode": "max",
        "stride": 2,
        "pad": "valid"
    }
)
    
flatten = AttrDict(
    {
        "name": "flatten"
    }
)


fc1 = AttrDict(
    {
        "name": "fully_connected",
        "activation": "relu",
        "weight_init": "he_uniform",
        "n_out": 120,
    }
)

fc2 = AttrDict(
    {
        "name": "fully_connected",
        "activation": "relu",
        "weight_init": "he_uniform",
        "n_out": 84,
    }
)

fc_out = AttrDict(
    {
        "name": "fully_connected",
        "activation": "softmax",  # Softmax for last layer for classification
        "weight_init": "he_uniform",
        "n_out": None
        # n_out is not defined for last layer. This will be set by the dataset.
    }
)

"""
Step 2: Collect layer argument dictionaries into a list.

- This defines the order of layers in the network.
"""

layer_args = [conv1, pool1, conv2, pool2, flatten, fc1, fc2, fc_out]

"""
Step 3: Define model, data, and logger arguments

- The list of layer_args is passed to the model initializer.
"""

optimizer_args = AttrDict(
    {
        "name": "SGD",
        "lr": 0.01,
        "lr_scheduler": "constant",
        "lr_decay": 0.99,
        "stage_length": 1000,
        "staircase": True,
        "clip_norm": 1.0,
        "momentum": 0.9,
    }
)

model_args = AttrDict(
    {
        "name": "feed_forward",
        "loss": "cross_entropy",
        "layer_args": layer_args,
        "optimizer_args": optimizer_args,
        "seed": 0,
    }
)

data_args = AttrDict(
    {
        "name": "mnist",
        "batch_size": 16,
    }
)

log_args = AttrDict(
    {"save": True, "plot": True, "save_dir": "experiments/",}
)

"""
Step 4: Set random seed

Warning! Random seed must be set before importing other modules.
"""

import numpy as np

np.random.seed(model_args.seed)

"""
Step 5: Define model name for saving
"""

model_name = "{}_{}layers_{}-lr{}_mom{}_seed{}".format(
    model_args.name,
    len(layer_args),
    fc1["n_out"],
    optimizer_args.lr,
    optimizer_args.momentum,
    model_args.seed,
)

"""
Step 6: Initialize logger, model, and dataset

- model_name, model_args, and data_args are passed to the logger for saving
- The logger is passed to the model.
"""

from neural_networks.models import initialize_model
from neural_networks.datasets import initialize_dataset
from neural_networks.logs import Logger


logger = Logger(
    model_name=model_name,
    model_args=model_args,
    data_args=data_args,
    save=log_args.save,
    plot=log_args.plot,
    save_dir=log_args.save_dir,
)


model = initialize_model(
    name=model_args.name,
    loss=model_args.loss,
    layer_args=model_args.layer_args,
    optimizer_args=model_args.optimizer_args,
    logger=logger,
)


dataset = initialize_dataset(
    name=data_args.name,
    batch_size=data_args.batch_size,
)


"""
Step 7: Train model!
"""

epochs = 5

print(
    "Training {} neural network on {} with {} for {} epochs...".format(
        model_args.name, data_args.name, optimizer_args.name, epochs
    )
)

print("Optimizer:")
print(optimizer_args)

model.train(dataset, epochs=epochs)