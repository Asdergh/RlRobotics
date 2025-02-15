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
from torchvision.io import write_video




class SimpleNet(Module):

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        hiden_features: int = 128
    ) -> None:
        
        super().__init__()
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
        return self._net_(inputs)


class SimpleTrainer:

    def __init__(
        self,
        target_path: str,
        model: Module,
        optim: Optimizer,
        env: gym.Env,
        discont_coeff: float = 12.12,
        save_scene: bool = True
    ) -> None:
        
        self.dis_coeff = discont_coeff
        self.epizode_tup = namedtuple("Epizode", [
            "LogMeanActions",
            "Observations",
            "Rewards",
            "Frames"
        ])
        
        self.losses = []
        self._save_scene_ = save_scene
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
        self.env.reset()
    

    def _train_on_epizode_(self, epizode: int, steps: int) -> tuple[
        float,
        namedtuple
    ]:

        
        epizode = self.epizode_tup([], [], [], [])
        action = env.action_space.sample()
        for _ in tqdm.tqdm(
            range(steps),
            desc="Trajectory Generation",
            colour="green",
            ascii=":>"
        ):

            if not isinstance(action, np.ndarray):
                action = action.detach().numpy()

            observations, reward, terminate = self.env.step(action)[:3]
            if terminate:
                pass

            action = self.model(th.Tensor(observations.copy()))

            epizode.LogMeanActions.append(th.log(th.sum(action)))
            epizode.Observations.append(observations)
            epizode.Rewards.append(reward)
            epizode.Frames.append(th.Tensor(env.render().copy()))
        
        ret = th.Tensor(epizode.Rewards).sum() * self.dis_coeff
        loss = th.stack([
            log_action.unsqueeze(dim=0) for log_action in epizode.LogMeanActions
        ], dim=0).sum() * ret
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return (loss, epizode)
    
    def train(self, epizodes: int, steps_per_epizode: int) -> namedtuple:
        
        for epizode in range(epizodes):
            
            self.env.reset()
            (loss, epizode_history) = self._train_on_epizode_(epizode=epizode, steps=steps_per_epizode)
            if self._save_scene_:
                
                frames = th.cat([
                    frame.unsqueeze(dim=0) for frame in epizode_history.Frames
                ], dim=0)
                path = os.path.join(
                    self._gen_folder_,
                    f"epizode_{epizode}"
                )
                if not os.path.exists(path):
                    os.mkdir(path)

                scene_path = os.path.join(path, "scene.mp4")
                rewards_path = os.path.join(path, "rewards.png")
                
                fig, axis = plt.subplots()
                rews = np.asarray(epizode_history.Rewards)
                axis.plot(rews, color="blue", linestyle="--")
                axis.fill_between(
                    np.linspace(0, rews.shape[-1], rews.shape[-1]), 
                    rews - np.mean(rews),
                    rews + np.mean(rews),
                    color="red", alpha=0.76
                )
                fig.savefig(rewards_path)
                del fig, axis

                write_video(
                    scene_path,
                    video_array=frames,
                    fps=30
                )
            
            del epizode_history
            params_path = os.path.join(self.target_path, "params.pt")
            self.losses.append(loss)
            th.save(self.model.state_dict(), params_path)
        
        return self.training_history
            
            

            
        
        
env = gym.make(
    'Humanoid-v5', 
    ctrl_cost_weight=0.1,
    render_mode="rgb_array",
    terminate_when_unhealthy=True
)


model = SimpleNet(
    n_observations=348,
    n_actions=17
)
optim = Adam(lr=0.01, params=model.parameters())
trainer = SimpleTrainer(
    model=model,
    optim=optim,
    env=env,
    target_path="C:\\Users\\1\\Desktop\\RLtarget"
)
trainer.train(
    epizodes=100,
    steps_per_epizode=1000
)

