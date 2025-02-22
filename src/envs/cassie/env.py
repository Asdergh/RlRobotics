import os
import yaml
import matplotlib.pyplot as plt
import torch as th
import mujoco as mj
import numpy as np
plt.style.use("seaborn")

from collections import namedtuple
from torchvision.io import write_video
import tqdm
import cv2




class CassieEnv:

    def __init__(self, params: dict | str) -> None:

        self.params = params
        self.memory = []
        self._sample_ = namedtuple("Sample", [
            "action",
            "observation",
            "reward"
        ])
        self._step_sample_ = namedtuple("StepSample", [
            "observation",
            "reward"
        ])
        
        if isinstance(params, str):
            with open(params, "r") as params_f:
                self.params = yaml.load(params_f)

        self.model = mj.MjModel.from_xml_path("C:\\Users\\1\\Desktop\\PythonProjects\\RL_robotics\\src\\envs\\cassie\\scene.xml")
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer(self.model)
        if self.params["render"]:
            self.frames = []
        
    
    
        
    def _get_linear_reward_(self, t_before, t_after) -> float:
        
        self.reset_status = True
        dx = (t_after - t_before)
        dt = self.params["timestep"]
        return self.params["linear_reward_weight"] * (dx / dt)

    def show_scene(self, path: str = None) -> None:
        
        assert self.params["render"], "Set render: bool = True, to collect frames samples!!!"
        for frame in self.frames:
            cv2.imshow("cassie_scene", cv2.resize(frame, (600, 600)))
            if cv2.waitKey(1) == ord('q'):
                break
        
    def plot_rewards(self) -> None:


        assert self.reset_status, "Can't plot rewards after reset. \nUse step to generation trajectory of samples!!!"
        _, axis = plt.subplots()
        rewards = np.asarray([sample.reward for sample in self.memory])
        axis.plot(
            rewards,
            color="blue",
            linestyle="--"
        )
        axis.fill_between(
            np.linspace(0, len(self.memory), len(self.memory)),
            rewards - rewards.mean(),
            rewards + rewards.mean(),
            color="gray",
            alpha=0.67,
        )
        
        plt.show()

        

    def step(self, action: np.ndarray) -> tuple:
    
        target_before = self.data.qpos[0]
        
        self.data.ctrl = action
        mj.mj_step(self.model, self.data)
        self.renderer.update_scene(self.data, camera="track")
        target_after = self.data.qpos[0]

        frame = self.renderer.render()
        
        linear_reward = self._get_linear_reward_(target_before, target_after)
        sample = self._sample_(
            action,
            th.cat([th.Tensor(self.data.qpos), th.Tensor(self.data.qvel)], dim=0),
            linear_reward
        )
        self.memory.append(sample)
        
        if self.params["render"]:
            self.frames.append(frame)
            return (sample, frame)
        
        else:
            return sample
    

    def reset(self) -> None:

        mj.mj_resetData(self.model, self.data)
        if not self.params["memory_based"]:
            self.memory = []
    

if __name__ == "__main__":
    
    test_env = CassieEnv(params={
        "timestep": 0.1,
        "linear_reward_weight": 1.0,
        "render": True,
        "memory_based": True
    })
    

    rewards = []
    frames = []
    for _ in tqdm.tqdm(
        range(1000),
        colour="green",
        ascii=":>"
    ):

        test_env.data.qpos[6] += 0.05
        action = np.random.normal(0.12, 5.12, test_env.model.nu)
        sample, frame = test_env.step(action)
        rewards.append(sample.reward)
        frames.append(frame)

    
    test_env.plot_rewards()
    test_env.show_scene()
    # test_env.save_rendered_scene()
    
    
    
    
    
        

    