import os
import random
import math

import os
'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
'''
import numpy as np
import magnum as mn
from PIL import Image
from agent.environment.settings import default_sim_settings, make_cfg
from habitat_sim.scene import SceneNode

import habitat_sim
import habitat_sim.agent
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_from_magnum,
    quat_to_magnum,
)


class HabitatEnv():
    def __init__(self, glb_path):
        self._cfg = make_cfg(glb_path)
        self._sim = habitat_sim.Simulator(self._cfg)
        random.seed(default_sim_settings["seed"])
        self._sim.seed(default_sim_settings["seed"])

        start_state = habitat_sim.agent.AgentState()
        start_state.position = np.array([0,0,0]).astype('float32')
        start_state.rotation = np.quaternion(1,0,0,0)
        self._sim.initialize_agent(default_sim_settings["default_agent"], start_state)
        self.observation = self.get_color_observation()

    def reset(self):
        self.observation = self.get_color_observation()
        # Implement Functionality to reset agent.
    
    def step(self, action):
        self._sim.step(action)
        self.observation = self.get_color_observation()

    def get_color_observation(self):
        obs = self._sim.get_sensor_observations()
        color_obs = obs["color_sensor"]
        return color_obs

    def save_color_observation(self, obs, frame, step, folder):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        color_img.save(folder + "/results/test.rgba.%05d.%05d.png" % (frame, step))
        color_img = read_gen(folder + "/results/test.rgba.%05d.%05d.png" % (frame, step))
        if self.depth_type == 'FLOW':
            if frame == 1:
                prev_color_img = read_gen(folder + "/results/test.rgba.%05d.%05d.png" % (frame-1, step-1))
                return color_img, prev_color_img
            elif frame > 1:
                prev_color_img = read_gen(folder + "/results/test.rgba.%05d.%05d.png" % (frame-1, step))
                return color_img, prev_color_img
            else:
                return color_img, color_img
        elif self.depth_type == 'TRUE':
            return color_img
    
    def save_depth_observation(self, obs, frame, step, folder):
        depth_obs = obs["depth_sensor"]
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        depth_img.save(folder + "/results/test.depth.%05d.%05d.png" % (frame, step))
        depth_img = plt.imread(folder + "/results/test.depth.%05d.%05d.png" % (frame, step))
        return depth_img


    def end_sim(self):
        self._sim.close()
        del self._sim