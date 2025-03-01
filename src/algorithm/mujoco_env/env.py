import os
import yaml
import sys
import matplotlib.pyplot as plt
import torch as th
import mujoco 
import numpy as np
plt.style.use("dark_background")

from collections import namedtuple 
from mujoco import (
    MjModel,
    MjData,
    Renderer
)


class MujocoEnv:

    def __init__(self, params: dict | str) -> None:
        
        self._root_ = os.path.dirname(os.path.abspath(__file__))
        self.params = params

        self._epizode_ = namedtuple("Epizode", [
            "Actions",
            "Observations",
            "Rewards",
        ])
        self._epizode_buffer_ = self._epizode_([], [], [])
        self._step_sample_ = namedtuple("StepSample", [
            "observation",
            "reward",
            "terminated"
        ])
        
        if isinstance(params, str):
            with open(params, "r") as params_f:
                self.params = yaml.load(params_f)


        _xml_ = os.path.join(
            self._root_,
            self.params["model_name"],
            "scene.xml"
        )
    
        self.model = MjModel.from_xml_path(_xml_)
        self.data = MjData(self.model)
        self.renderer = Renderer(self.model)

        self._epizode_idx_ = 0
        
    

    def _get_reward_(self, t_before, t_after) -> float:
        
        self.reset_status = True
        dx = (t_after - t_before)
        dtheta = -np.linalg.norm(self.data.qpos[5:7])
        dt = self.params["timestep"]
        linear_reward = self.params["linear_reward_weight"] * (dx / dt)
        anguler_reward = self.params["anguler_reward_weight"] * (dtheta / dt)

        return linear_reward + anguler_reward
    
        

    def step(self, action: np.ndarray) -> tuple:

        target_before = self.data.qpos[0]
        
        self.data.ctrl += action
        if (self.data.qpos[2] < 0.5):
            mujoco.mj_resetData(self.model, self.data)

        mujoco.mj_step(self.model, self.data)
        self.renderer.update_scene(self.data)
        target_after = self.data.qpos[0]

        reward = self._get_reward_(target_before, target_after)
        observation = th.cat([th.Tensor(self.data.qpos), th.Tensor(self.data.qvel)], dim=0)
        
        self._epizode_buffer_.Actions.append(action)
        self._epizode_buffer_.Observations.append(observation)
        self._epizode_buffer_.Rewards.append(reward)

        epizode_end = (self.data.qpos[2] <= 0.5)
        out_sample = self._step_sample_(
            observation,
            reward,
            epizode_end
        )
        

        if not self.params["render"]:
            return out_sample
        
        else:
            frame = self.renderer.render()
            return (out_sample, frame)
    

    def reset(self) -> None:
        
        mujoco.mj_resetData(self.model, self.data)
        if self.params["memory_base"]:
            self.memory[str(self._epizode_idx_)] = self._epizode_buffer_
        
        else:
            self.memory = self._epizode_buffer_
        
        self._epizode_buffer_ += 1
        
        
    


if __name__ == "__main__":

    test_env = MujocoEnv({
        "render": True,
        "linear_reward_weight": 1,
        "anguler_reward_weight": 1,
        "memory_based": True,
        "timestep": 0.003,
        "model_name": "g1_robot"
    })

    print(test_env.data.qpos.shape, test_env.data.qvel.shape)
    

    
    
    
    
    
        

    