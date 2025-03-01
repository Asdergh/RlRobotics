import gymnasium as gym
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import sys
plt.style.use("seaborn")

from torch.optim import Optimizer, Adam
from torch.nn import (
    Module,
    Sequential,
    Linear,
    ReLU,
    LayerNorm,
    Tanh,
    Dropout
)
from base.base_alg import Algorithm


class SimpleNet(Module):

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        hiden_features: int = 512,
        in_set: tuple[int] = (-1, 1),
        out_set: tuple[int] = (-0.4, 0.4)
    ) -> None:
        
        super().__init__()
        self._in_set_ = in_set
        self._out_set_ = out_set
        self._scale_coeff_ = (max(out_set) - min(out_set)) / (max(in_set) - min(in_set))
        self._net_ = Sequential(
            Linear(in_features=n_observations, out_features=hiden_features),
            Tanh(),
            Dropout(p=0.45),
            LayerNorm(normalized_shape=hiden_features),
            Linear(in_features=hiden_features, out_features=n_actions),
            Tanh(),
            Dropout(p=0.45),
            LayerNorm(normalized_shape=n_actions),
            Tanh()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        inputs = (inputs - inputs.mean()) / inputs.std()
        return (self._scale_coeff_ * (self._net_(inputs) - min(self._in_set_))) + min(self._out_set_)



class REINFORCE(Algorithm):

    def __init__(self, net, params):
        super().__init__(net, params)
    

    def train_on_epizode(self, epizode: int) -> tuple[float]:

        self._optim_.zero_grad()
        false_n = 0
        action = np.random.normal(0.0, 1.0, self._env_.model.nu)
        for _ in range(self.params["alg"]["steps"]):
            
            if false_n >= self.params["alg"]["false_tallerance"]:
                break

            if self.params["capture_scene"]:
                (observation, _, terminate), frame = self._env_.step(action)
                self.frames.append(frame)

            else:
                observation, _, terminate = self._env_.step(action)
                
            
            action = self._net_(observation).detach().numpy()
            if terminate:
                false_n += 1
        
        # for step in range(self.params["alg"]["steps"]):

        rets = sum([
            (self.params["alg"]["gamma"] ** i) * reward
            for (i, reward) in enumerate(self._env_._epizode_buffer_.Rewards)
        ])
        observations = th.Tensor(np.stack(self._env_._epizode_buffer_.Observations, axis=0))
        pd_actions = self._net_(observations)
    

        loss = th.sum(rets * -pd_actions)
        loss.backward()
        self._optim_.step()

        self.losses.append(loss.item())
        self.rewards.append(np.asarray(self._env_._epizode_buffer_.Rewards).mean())

    def train(self) -> None:

        

        for epizode in tqdm.tqdm(
            range(self.params["alg"]["epizodes"]),
            colour="green",
            ascii=":>",
            desc="Traning"
        ):
            self.train_on_epizode(epizode=epizode)
        
        self.losses = np.asarray(self.losses)
        self.rewards = np.asarray(self.rewards)
        self.plot_stats()
        self.show_scene()

        self.save_weights()

        



params = {
    "capture_scene": True,
    "env": {
        "type": "mujoco",
        "params": {
            "render": True,
            "linear_reward_weight": 1,
            "anguler_reward_weight": 1,
            "memory_based": True,
            "timestep": 0.003,
            "model_name": "cassie"
        }
    },
    "opt": {
        "type": "adam",
        "learning_rate": 0.01
    },
    "alg": {
        "type": "reinforce",
        "gamma": 0.45,
        "false_tallerance": 1,
        "epizodes": 100,
        "steps": 100
    }
}
# env = MujocoEnv({
#     "render": False,
#     "linear_reward_weight": 1,
#     "anguler_reward_weight": 0,
#     "memory_based": True,
#     "timestep": 0.003,
#     "model_name": "cassie"
# })

model = SimpleNet(
    n_observations=(35 + 32),
    n_actions=10
)
alg = REINFORCE(net=model, params=params)
alg.train()




