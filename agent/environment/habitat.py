# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import random
from PIL import Image
from agent.environment.environment import Environment
from agent.environment.habitatenv import HabitatEnv, e2q, q2e
from torchvision import transforms

def photometric_error(img1, img2):
    imageA = np.asarray(img1)
    imageB = np.asarray(img2) 
    err = np.sum((imageA.astype("float") - imageB.astype("float"))**2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

'''
def pose_error(p, gt_p, yaw):
    px = abs(gt_p[0] - p.position[0])
    py = abs(gt_p[1] - p.position[1])
    pz = abs(gt_p[2] - p.position[2])
    dyaw = abs(yaw - q2e(p.rotation.w, p.rotation.x, p.rotation.y, p.rotation.z)[1])
    #return px,py,pz
    #return px,pz,dyaw
    return px,py,pz
'''

def pose_error(p, gt_p):
    px = abs(gt_p[0] - p.position[0])
    py = abs(gt_p[1] - p.position[1])
    pz = abs(gt_p[2] - p.position[2])
    return px,py,pz


class HabitatDiscreteEnvironment(Environment):
    def __init__(self,
            scene_name = 'Arkansaw',
            screen_width = 384,
            screen_height = 512,
            terminal_image = None,
            history_length = 4,
            **kwargs):
        super(HabitatDiscreteEnvironment, self).__init__()
        self.env = HabitatEnv(scene_name)
        self.history_length = history_length
        target_image = Image.open(terminal_image)
        target_image = np.array(target_image, dtype=np.float32)[:,:,:3] / 255
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.target = (target_image - self.mean) / self.std
        self.target = np.repeat(self.target[np.newaxis,...], self.history_length, axis=0)
        self.target_pose = [ -0.05, 0, -0.5, 1, 0, 0, 0]
        self.yaw = np.degrees(q2e(self.target_pose[3], self.target_pose[4], self.target_pose[5], self.target_pose[6])[1])
        self.collision = False

    def reset(self):
        self.time = 0
        self.collided = 0
        self.terminal = 0
        self.collision = False

        self.env.reset()
        curr_image = self.env.get_color_observation()
        curr_image = (curr_image - self.mean) / self.std
        self.state = np.repeat(curr_image[np.newaxis,...], self.history_length, axis=0)

    def step(self, action):
        assert not self.terminal, 'step() called in terminal state'
        self.env.step(self.actions[action])
        '''
        pe = photometric_error(self.state[-1], self.target[-1])
        if pe < 30000:
            print(pe)
            self.tminal = True
            # save image
            self.env.save_image()
        '''

        curr_pose = self.env.get_agent_pose()
        #px,pz,dyaw = pose_error(curr_pose, self.target_pose, self.yaw)
        print(curr_pose)
        px,py,pz = pose_error(curr_pose, self.target_pose)
        print(px)
        print(py)
        print(pz)

        '''
        if px <= 0.1 and pz <= 0.1 and dyaw < 5:
            self.terminal = True
            self.env.save_image()
        '''

        if px <= 0.1 and py <= 0.1 and pz <= 0.1:
            self.terminal = True
            self.env.save_image()
        
        if px > 1.0 or pz > 1.0 or py > 1.0:
            self.collision = True
        
        self.env.save_image_all()

        #self.s_t = np.append(self.s_t[:,1:], self._get_state(self.current_state_id), axis=1)
        # Remove last image and append new.
        self.state[:-1] = self.state[1:]
        curr_image = self.env.get_color_observation()
        curr_image = (curr_image - self.mean) / self.std
        self.state[-1] = curr_image
        self.time += 1

    def get_agent_pose(self):
        return self.env.get_agent_pose()
        
    def render(self):
        return self.state
    
    def render_target(self):
        return self.target

    def _calculate_reward(self, terminal, collided):
        # positive reward upon task completion
        if terminal: return 10.0
        # time penalty or collision penalty
        return -0.1

    @property
    def reward(self):
        return self._calculate_reward(self.is_terminal, self.collided)

    @property
    def is_terminal(self):
        return self.terminal or self.time >= 600
    
    @property
    def is_collision(self):
        return self.collision

    @property
    def actions(self):
        '''
        return ["PositiveSurge", "NegativeYaw", "PositiveYaw"]
        '''
        return ["PositiveSurge", "NegativeSurge", "PositiveHeave", "NegativeHeave",
                "PositivePitch", "NegativePitch", "PositiveYaw", "NegativeYaw"]
        return ["PositiveSurge", "PositiveSway", "PositiveHeave",
                "NegativeSway", "NegativeHeave", "NegativeSurge"]

        '''
        return ["PositiveSurge", "PositiveSway", "NegativeSway", 
                ""]
        '''
        '''
        return ["PositiveSurge", "PositiveSway", "PositiveHeave", 
                "PositiveRoll", "PositivePitch", "PositiveYaw",
                "NegativeSway", "NegativeHeave", "NegativeRoll", 
                "NegativePitch", "NegativeYaw", "NegativeSurge"]
        '''
'''
class THORDiscreteEnvironment(Environment):
    @staticmethod
    def _get_h5_file_path(h5_file_path, scene_name):
        if h5_file_path is None:
            h5_file_path = f"/app/data/{scene_name}.h5"
        elif callable(h5_file_path):
            h5_file_path = h5_file_path(scene_name)
        return h5_file_path

    def __init__(self, 
            scene_name = 'bedroom_04',
            n_feat_per_location = 1,
            history_length : int = 4,
            screen_width = 224,
            screen_height = 224,
            terminal_state_id = 0,
            initial_state_id = None,
            h5_file_path = None,
            **kwargs):
        super(THORDiscreteEnvironment, self).__init__()



        h5_file_path = THORDiscreteEnvironment._get_h5_file_path(h5_file_path, scene_name)
        self.terminal_state_id = terminal_state_id
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.n_feat_per_location = n_feat_per_location
        self.locations = self.h5_file['location'][()]
        self.rotations = self.h5_file['rotation'][()]
        self.history_length = history_length
        self.n_locations = self.locations.shape[0]
        self.terminals = np.zeros(self.n_locations)
        self.terminals[terminal_state_id] = 1
        self.terminal_states, = np.where(self.terminals)
        self.transition_graph = self.h5_file['graph'][()]
        self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self.initial_state_id = initial_state_id
        self.s_target = self._tiled_state(self.terminal_state_id)
        self.time = 0

    def _get_graph_handle(self):
        return self.h5_file

    def get_initial_states(self, goal):
        initial_states = list()
        for k in range(self.n_locations):
            min_d = self.shortest_path_distances[k][goal]
            if min_d > 0:
                initial_states.append(k)

        return initial_states

    @staticmethod
    def get_existing_initial_states(scene_name = 'bedroom_04',
        terminal_state_id = 0,
        h5_file_path = None):
        env = THORDiscreteEnvironment(h5_file_path, scene_name)
        return env.get_existing_initial_states(terminal_state_id)

    def reset(self, initial_state_id = None):
        # randomize initial state
        if initial_state_id is None:
            initial_state_id = self.initial_state_id

        if initial_state_id is None:
            while True:
                k = random.randrange(self.n_locations)
                min_d = np.inf

                # check if target is reachable
                for t_state in self.terminal_states:
                    dist = self.shortest_path_distances[k][t_state]
                    min_d = min(min_d, dist)

                # min_d = 0  if k is a terminal state
                # min_d = -1 if no terminal state is reachable from k
                if min_d > 0: break
        else:
            k = initial_state_id
        
        # reset parameters
        self.current_state_id = k
        self.s_t = self._tiled_state(self.current_state_id)

        self.collided = False
        self.terminal = False
        self.time = 0

    def step(self, action):
        assert not self.terminal, 'step() called in terminal state'
        k = self.current_state_id
        if self.transition_graph[k][action] != -1:
            self.current_state_id = self.transition_graph[k][action]
            if self.terminals[self.current_state_id]:
                self.terminal = True
                self.collided = False
            else:
                self.terminal = False
                self.collided = False
        else:
            self.terminal = False
            self.collided = True

        self.s_t = np.append(self.s_t[:,1:], self._get_state(self.current_state_id), axis=1)
        self.time = self.time + 1

    def _get_state(self, state_id):
        # read from hdf5 cache
        k = random.randrange(self.n_feat_per_location)
        return self.h5_file['resnet_feature'][state_id][k][:,np.newaxis]

    def _tiled_state(self, state_id):
        f = self._get_state(state_id)
        return np.tile(f, (1, self.history_length))

    def _calculate_reward(self, terminal, collided):
        # positive reward upon task completion
        if terminal: return 10.0
        # time penalty or collision penalty
        return -0.1 if collided else -0.01

    @property
    def reward(self):
        return self._calculate_reward(self.is_terminal, self.collided)

    @property
    def is_terminal(self):
        return self.terminal or self.time >= 5e3

    @property
    def actions(self):
        return ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward"]
'''
