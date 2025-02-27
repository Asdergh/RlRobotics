import abc
import os
import json as js
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple


class BaseEnv(abc.ABC):


    def __init__(self, params: tuple | str) -> None:

        self.params = params
        if isinstance(self.params, str):
            with open(self.params, "r") as params_path:
                self.params = js.load(params_path)
            
        self._epizode_ = namedtuple("Epizode", [
            "Actions",
            "Rewards",
            "Observations"
        ])
        self._step_sample_ = namedtuple("StepSample", [
            "observation",
            "reward",
            "terminated"
        ])

        if self.params["render"]:
            if self.params["memory_based"]:
                self.frames = {}
            self._frames_epizode_ = []
        

        if self.params["memory_based"]:
            self.memory = {}
            self._epizode_idx_ = 0
        
        else:
            self.memory = []
    

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def reset():
        pass

    def plot_rewards(self, idx: int) -> None:

        
        _, axis = plt.subplots()
        if not self.params["memory_based"]:
            axis.plot(self.memory, color=np.random.randomt(3), linestyle="--")

        else:
            
            for idx in range(self._epizode_idx_):
                ax = axis.twinx()
                ax.grid(True)
                ax.plot()
            
            
        