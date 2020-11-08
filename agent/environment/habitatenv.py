import os
import random
import math

import cv2

import os
'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
'''
import numpy as np
import magnum as mn
from PIL import Image
from agent.environment.settings import default_sim_settings, make_cfg
# uncomment while generating image
#from settings import default_sim_settings, make_cfg
from habitat_sim.scene import SceneNode

import habitat_sim
import habitat_sim.agent
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_from_magnum,
    quat_to_magnum,
)

def pose_error(p, gt_p):
    px = abs(gt_p[0] - p[0])
    py = abs(gt_p[1] - p[1])
    pz = abs(gt_p[2] - p[2])
    return px,py,pz

def e2q(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qw, qx, qy, qz]

def q2e(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z 

class HabitatEnv():
    def __init__(self, glb_path):
        self._cfg = make_cfg(glb_path)
        self._sim = habitat_sim.Simulator(self._cfg)
        random.seed(default_sim_settings["seed"])
        self._sim.seed(default_sim_settings["seed"])

        self.start_state = habitat_sim.agent.AgentState()
        self.start_state.position = np.array([0,0,0]).astype('float32')
        self.start_state.rotation = np.quaternion(1,0,0,0)
        self.agent = self._sim.initialize_agent(default_sim_settings["default_agent"], self.start_state)
        self.count = 0

    def reset(self):
        tp = [ -0.05, 0, -0.5, 1, 0, 0, 0]
        px = 4 
        py = 4
        pz = 4
        temp_new = None
        while px > 1 and py > 1 and pz > 1:
            print("in while loop reset")
            temp_new = np.random.rand(3) * 0.01
            px,py,pz = pose_error(tp[0:3], temp_new)
        
        self.start_state.position = temp_new

        #q = e2q(0, np.random.rand(1)*10, 0)
        #self.start_state.rotation = np.quaternion(q[0], q[1], q[2], q[3])
        self.start_state.rotation = np.quaternion(1, 0, 0, 0) 
        self.agent.set_state(self.start_state)
    
    def step(self, action):
        self._sim.step(action)
        self.count += 1

    def save_image(self):
        cv2.imwrite("/scratch/rl/rgb.%05d.png" % self.count, self.get_color_observation()[..., ::-1])
    
    def save_image_all(self):
        cv2.imwrite("/scratch/rl_all/rgb.%05d.png" % self.count, self.get_color_observation()[..., ::-1])
    
    def goto_state(self, state):
        self.start_state.position = np.array([state[0], state[1], state[2]]).astype('float32')
        self.start_state.rotation = np.quaternion(state[3], state[4], state[5], state[6])
        self.agent.set_state(self.start_state)
        cv2.imwrite("/scratch/rl/rgb.%05d.png" % self.count, self.get_color_observation()[..., ::-1])

    def get_color_observation(self):
        obs = self._sim.get_sensor_observations()
        color_obs = obs["color_sensor"]
        image = np.array(color_obs[:,:,:3])
        return image
    
    def get_agent_pose(self):
        state = self.agent.get_state()
        return state

    def end_sim(self):
        self._sim.close()
        del self._sim
