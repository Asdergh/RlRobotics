import numpy as np
import torch as th
import mujoco 
import json as js
import abc
import cv2
import os
import matplotlib.pyplot as plt

from torch.nn import Module
from mujoco_env.env import MujocoEnv
from torch.optim import (
    Adam,
    SGD
)


__envs__ = {
    "mujoco": MujocoEnv
}
__optimizers__ = {
    "adam": Adam,
    "sgd": SGD
}

class Algorithm(abc.ABC):

    def __init__(
        self,
        net: Module,
        params: dict | str
    ) -> None:
        

        self.rewards = []
        self.losses = []
        self.net = net
        self.params = params
        if isinstance(self.params, str):
            with open(self.params, "r") as params_f:
                self.params = js.load(params_f)
        
        
        if self.params["capture_scene"]:
            self.frames = []

        self._env_ = __envs__[self.params["env"]["type"]](self.params["env"]["params"])
        self._net_ = net
        self._optim_ = __optimizers__[self.params["opt"]["type"]](
            lr=self.params["opt"]["learning_rate"],
            params=self._net_.parameters()
        )
    

    @abc.abstractmethod
    def train(self, epizodes: int, steps: int) -> None:
        pass

    @abc.abstractmethod
    def train_on_epizode(self, epizode: int, steps: int):
        pass

    def show_scene(self) -> None:

        for frame in self.frames:    
            cv2.imshow("traning_scene", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    
    def save_weights(self) -> None:
        
        weights_path = os.path.join(
            os.path.dirname(".."),
            "weights",
            f"{self.params['agent_name']}.pt"
        )
        th.save(weights_path, self._net_.state_dict())
    
    def plot_stats(self) -> None:

        print(self.losses.max(), self.losses.min())
        _, axis = plt.subplots()

        axis.plot(self.losses, color="orange", linestyle="--")
        axis.fill_between(
            np.linspace(0, self.losses.shape[0], self.losses.shape[0]),
            self.losses - self.losses.mean(),
            self.losses + self.losses.mean(),
            color="orange",
            alpha=0.67
        )

        rewards_plot = axis.twinx()
        rewards_plot.plot(self.rewards, color="blue", linestyle="-")
        rewards_plot.fill_between(
            np.linspace(0, self.losses.shape[0], self.losses.shape[0]),
            self.rewards - self.rewards.mean(),
            self.rewards + self.rewards.mean(),
            color="blue",
            alpha=0.67
        )

        plt.show()


        
        


        