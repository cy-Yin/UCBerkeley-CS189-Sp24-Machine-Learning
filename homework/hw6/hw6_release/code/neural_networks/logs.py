"""
Author: Sophia Sanborn
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class Logger:
    def __init__(
        self,
        model_name,
        model_args,
        data_args,
        save=False,
        plot=False,
        save_dir="experiments/",
    ):

        self.model_name = model_name
        self.model_args = model_args
        self.data_args = data_args
        self.save = save
        self.save_dir = save_dir + model_name + "/"
        self.plot = plot
        self.counter = 0
        self.log = {}

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        with open(self.save_dir + "model_args", "wb") as f:
            pickle.dump(self.model_args, f)

        with open(self.save_dir + "data_args", "wb") as f:
            pickle.dump(self.data_args, f)

    def push(self, log):
        if self.counter == 0:
            self.log = {k: {} for k in log.keys()}

            # self.log = {k: [] if k != "params" else {} for k in log.keys()}
            if "params" in log.keys():

                self.log["params"] = {
                    k: {"max": [], "min": []} for k in log["params"].keys()
                }

            self.log["loss"] = {"train": [], "validate": []}
            self.log["error"] = {"train": [], "validate": []}

        self.counter += 1
        for k, v in log.items():
            if k == "params":
                for param, vals in v.items():
                    self.log["params"][param]["max"].append(vals["max"])
                    self.log["params"][param]["min"].append(vals["min"])

            else:
                self.log[k]["train"].append(v["train"])
                self.log[k]["validate"].append(v["validate"])

        if self.save:
            with open(self.save_dir + "log", "wb") as f:
                pickle.dump(self.log, f)
            if self.plot:
                self._plot()

    def reset(self):
        self.log = {}
        self.counter = 0

    def _plot(self):
        for k, v in self.log.items():
            if k == "params":
                for param, vals in v.items():
                    plt.figure(figsize=(15, 10))
                    plt.plot(vals["max"], label="{}_max".format(param))
                    plt.plot(vals["min"], label="{}_min".format(param))
                    plt.legend()
                    plt.xlabel("epochs")
                    plt.ylabel(param)
                    plt.title(self.model_name)
                    plt.savefig(self.save_dir + param)
                    plt.close()
            else:
                plt.figure(figsize=(15, 10))
                plt.plot(v["train"], label="training")
                plt.plot(v["validate"], label="validation")
                plt.legend()
                plt.xlabel("epochs")
                plt.ylabel(k)
                plt.title(self.model_name)
                plt.savefig(self.save_dir + k)
                plt.close()
