import os
import yaml
import matplotlib.pyplot as plt
import torch as th
import mujoco as mj
import numpy as np
plt.style.use("dark_background")

from collections import namedtuple 
import cv2




class CassieEnv:

    def __init__(self, params: dict | str) -> None:

        self.params = params
        self._ENV_TYPE_ = "nondiscrite"
        
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

        self.model = mj.MjModel.from_xml_path("C:\\Users\\1\\Desktop\\PythonProjects\\RL_robotics\\src\\envs\\cassie\\scene.xml")
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer(self.model)
        
        
        if self.params["render"]:
            if self.params["memory_based"]:
                self.frames = {}

            self._frames_buffer_ = []

        if self.params["memory_based"]:
            self.memory = {}
            
            self._epizode_idx_ = 0
        
    
    
        
    def _get_linear_reward_(self, t_before, t_after) -> float:
        
        self.reset_status = True
        dx = (t_after - t_before)
        dt = self.params["timestep"]
        return self.params["linear_reward_weight"] * (dx / dt)


    def show_scene(self, ep_number: int) -> None:
        
        assert self.params["render"], "Set render: bool = True, to collect frames samples!!!"
        assert ep_number < max([int(idx) for idx in list(self.frames.keys())]), f"There is no epizode with id: {ep_number}"

        if not self.params["memory_based"]:
            frames = self.frames
        
        else:
            frames = self.frames[str(ep_number)]
        
        for frame in frames:
            cv2.imshow("cassie_scene", cv2.resize(frame, (600, 600)))
            if cv2.waitKey(1) == ord('q'):
                break
        

    def plot_rewards(self) -> None:


        assert self.reset_status, "Can't plot rewards after reset. \nUse step to generation trajectory of samples!!!"
        if not self.params["memory_based"]:

            _, axis = plt.subplots()
            rewards = np.asarray(self._epizode_buffer_.Rewards)
            axis.plot(
                rewards,
                color="blue",
                linestyle="--",
                label="rewards"
            )
            axis.fill_between(
                np.linspace(0, rewards.shape[0], rewards.shape[0]),
                rewards - rewards.mean(),
                rewards + rewards.mean(),
                color="gray",
                alpha=0.67,
            )

        else:

            _, axis = plt.subplots()
            for (ep_idx, ep) in self.memory.items():
                
                rewards = np.asarray(ep.Rewards)
                color = np.random.random(3)
                ax = axis.twinx()
                ax.grid(False)
                ax.plot(
                    rewards,
                    color=color,
                    linestyle="--",
                    label=f"{ep_idx}[rewards]"
                )
                ax.fill_between(
                    np.linspace(0, rewards.shape[0], rewards.shape[0]),
                    rewards - rewards.mean(),
                    rewards + rewards.mean(),
                    color=color,
                    alpha=0.67,
                )
            
        axis.legend(loc="upper left")
        plt.show()


    def step(self, action: np.ndarray) -> tuple:

        target_before = self.data.qpos[0]
        
        self.data.ctrl += action
        if (self.data.qpos[2] < 0.5):
            mj.mj_resetData(self.model, self.data)

        mj.mj_step(self.model, self.data)
        self.renderer.update_scene(self.data, camera="track")
        target_after = self.data.qpos[0]

        frame = self.renderer.render()
        
        linear_reward = self._get_linear_reward_(target_before, target_after)
        observation = th.cat([th.Tensor(self.data.qpos), th.Tensor(self.data.qvel)], dim=0)
        
        self._epizode_buffer_.Actions.append(action)
        self._epizode_buffer_.Observations.append(observation)
        self._epizode_buffer_.Rewards.append(linear_reward)

        epizode_end = (self.data.qpos[2] <= 0.5)
        out_sample = self._step_sample_(
            observation,
            linear_reward,
            epizode_end
        )
        if self.params["render"]:
            self._frames_buffer_.append(frame)

        return out_sample
    

    def reset(self) -> None:
        
        mj.mj_resetData(self.model, self.data)
        if self.params["memory_based"]:
            self.memory[str(self._epizode_idx_)] = self._epizode_buffer_
            self.frames[str(self._epizode_idx_)] = self._frames_buffer_
        
        self._epizode_buffer_ = self._epizode_([], [], [])
        self._frames_buffer_ = []
        self._epizode_idx_ += 1
        
    



    
    
    
    
    
        

    