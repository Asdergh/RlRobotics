import torch as th
import gymnasium as gym
import tqdm 
import numpy as np

from collections import namedtuple
from torch.nn import (
    Sequential,
    Linear,
    Module,
    Dropout,
    Softmax,
    Sigmoid,
    ReLU,
)
from torch.optim import (
    Optimizer,
    Adam
)
import matplotlib.pyplot as plt
import cv2




# env ------------------------------------------------
env = gym.make("LunarLander-v3", continuous=False, render_mode="rgb_array")
env.reset()
#-----------------------------------------------------


# model ----------------------------------------------
class QNet(Module):

    def __init__(
        self,
        ob_dim: int,
        action_dim: int,
        hiden_dim: int = 128,
    ) -> None:
        
        super().__init__()
        self._net_ = Sequential(
            Linear(ob_dim , hiden_dim),
            Linear(hiden_dim, hiden_dim),
            Linear(hiden_dim, action_dim),
            Softmax(dim=1)
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net_(inputs)
#------------------------------------------------------

# algorithm -------------------------------------------

class SARSA:

    def __init__(
        self,
        model: Module,
        env: gym.Env,
        optim: Optimizer,
        epsilon: float = 1.0,
        dis_coeff: float = 0.78,
    ) -> None:
        
        self.net = model
        self.env = env
        self.opt = optim
        self._gamma_ = dis_coeff
        self._epsilon_ = epsilon

        self._rets_ = []
        self._losses_ = []

        
        self._epizode_sample_ = namedtuple("Epizode", [
            "state_p",
            "action_p",
            "reward",
            "state_a",
            "action_a"
        ])
        self._training_frames_ = {}
        
    

    def train(self, epizodes: int, steps: int) -> None:
        
        losses = []
        rewards = []
        for epizode in range(epizodes):
            loss, mean_reward = self._train_on_epizode_(epizode=epizode, steps=steps)
            losses.append(loss)
            rewards.append(mean_reward)
            self._epsilon_ -= 0.4
        
        losses = np.asarray(losses)
        rewards = np.asarray(rewards)

        _, axis = plt.subplots()
        time = np.linspace(0, epizodes, epizodes)
        axis.plot(losses, color="blue", linestyle="--")
        axis.fill_between(
            time, 
            losses - losses.mean(), 
            loss + losses.mean(), 
            alpha=0.45, 
            color="green"
        )

        rewards_ax = axis.twinx()
        rewards_ax.plot(rewards, color="blue", linestyle="--")
        rewards_ax.fill_between(
            time, 
            rewards - rewards.mean(), 
            rewards + rewards.mean(), 
            alpha=0.45, 
            color="red"
        )
        plt.show()

        for step in range(steps):

            frames_stack = np.concatenate([
                cv2.resize(self._training_frames_[f"{idx}"][step], (600, 600))
                for idx in range(epizodes)
            ], axis=1)
            cv2.imshow("scene", frames_stack)
            if cv2.waitKey(1) == ord("q"):
                break

        
    def _train_on_epizode_(self, epizode: int, steps: int) -> tuple[float]:

        self._training_frames_[f"{epizode}"] = []
        trajectory = []
        action = int(th.rand(self.env.action_space.n).argmax().item())
        for idx in tqdm.tqdm(
            range(steps),
            colour="green",
            ascii=":>",
            desc=f"Generating Trajectory; Ep: [{epizode}]"
        ):
            
            observ_p, reward = self.env.step(action)[:2]
            if self._epsilon_ > th.rand(1).item():
                action = self.env.action_space.sample()
            
            else:

                out = self.net(th.Tensor(observ_p).unsqueeze(dim=0))
                action = int(out.argmax().item())
            
            
            self._training_frames_[f"{epizode}"].append(self.env.render())            
            observ_a = self.env.step(action)[0]
            action_a = int(self.net(th.Tensor(observ_a).unsqueeze(dim=0)).argmax())

            trajectory.append(self._epizode_sample_(
                observ_p,
                action,
                reward,
                observ_a,
                action_a
            ))
        
        
        labels = []
        target_labels = []
        for (ob_p, _, rew, ob_a, _) in trajectory:

            ob_a = th.Tensor(ob_a).unsqueeze(dim=0)
            ob_p = th.Tensor(ob_p).unsqueeze(dim=0)
            tar_label = rew + (self._gamma_ * self.net(ob_a).max())
            target_labels.append(tar_label)

            #TODO correct SARSA alg.
        
        self.opt.zero_grad()
        labels = th.Tensor(labels)
        target_labels = th.Tensor(target_labels)
        loss = (labels - target_labels)
        loss.backward()
        self.opt.step()
        
        return (loss.mean(), np.asarray([sample.reward for sample in trajectory]).mean())
    



if __name__ == "__main__":
    

    model = QNet(
        action_dim=4,
        ob_dim=8,
        hiden_dim=128
    )
    optim = Adam(lr=0.01, params=model.parameters())
    alg = SARSA(
        env=env,
        optim=optim,
        model=model
    )
    
    # alg._train_on_epizode_(epizode="[exp. epizode]", steps=10)
    alg.train(
        epizodes=3,
        steps=3000
    )


            
            
            
            
        
        

        

    

    
