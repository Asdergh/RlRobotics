import gymnasium as gym
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
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
from collections import namedtuple
from envs.cassie.env import CassieEnv




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


class SimpleTrainer:

    def __init__(
        self,
        target_path: str,
        model: Module,
        optim: Optimizer,
        env: CassieEnv,
        discont_coeff: float = 0.12,
    ) -> None:
        
        self.dis_coeff = discont_coeff
        self.epizode_tup = namedtuple("Epizode", [
            "LogMeanActions",
            "Observations",
            "Rewards",
        ])
        
        self.losses = []
        self.target_path = target_path
        self._gen_folder_ = os.path.join(
            self.target_path,
            "gen_folder"
        )
        for path in [self.target_path, self._gen_folder_]:
            if not (os.path.exists(path)):
                os.mkdir(path)

        self.model = model
        self.optim = optim
        self.env = env
        # self.env.reset()
    

    def _train_on_epizode_(self, ep_idx: int) -> tuple[float, namedtuple]:

        
        epizode = self.epizode_tup([], [], [])
        action = np.zeros(self.env.data.ctrl.shape[0])
        epizode_end = False
        while not epizode_end:

            if not isinstance(action, np.ndarray):
                action = action.detach().numpy()

            observations, reward, terminate = self.env.step(action)
            action = self.model(observations)
            epizode.LogMeanActions.append(th.log(th.sum(action)))
            epizode.Observations.append(observations)
            epizode.Rewards.append(reward)

            epizode_end = terminate
        
        ret = th.Tensor([rew * (self.dis_coeff ** t) for (t, rew) in enumerate(epizode.Rewards)]).sum()
        loss = th.stack([
            log_action.unsqueeze(dim=0) for log_action in epizode.LogMeanActions
        ], dim=0).sum() * ret
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


        return (loss, epizode)
    
    def train(self, epizodes: int) -> None:
        
        for epizode in tqdm.tqdm(
            range(epizodes),
            colour="green",
            ascii=":>",
            desc="Training"
        ):
            
            (loss, epizode_history) = self._train_on_epizode_(ep_idx=epizode)

            params_path = os.path.join(self.target_path, "params.pt")
            self.losses.append(loss)

            th.save(self.model.state_dict(), params_path)
            self.env.reset()
        
        th.save(
            self.model.state_dict(),
            "C:\\Users\\1\\Desktop\\PythonProjects\\RL_robotics\\models_weights\\vanila_reinforce.pt"
        )
        
        

env = CassieEnv({
    "render": True,
    "linear_reward_weight": 1,
    "memory_based": True,
    "timestep": 0.003
})
print(env.data.qpos.shape)
print(env.data.qvel.shape)
model = SimpleNet(
    n_observations=(35 + 32),
n_actions=10
)
optim = Adam(lr=0.01, params=model.parameters())
trainer = SimpleTrainer(
    model=model,
    optim=optim,
    env=env,
    target_path="C:\\Users\\1\\Desktop\\RLtarget",
    discont_coeff=1.0
)
trainer.train(epizodes=200)
env.plot_rewards()

for idx in range(199):
    env.show_scene(idx)

