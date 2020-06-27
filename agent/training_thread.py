from agent.network import SceneSpecificNetwork, SharedNetwork, ActorCriticLoss
from agent.environment import Environment, HabitatDiscreteEnvironment
import torch.nn as nn
from typing import Dict, Collection
import signal
import random
import torch
from agent.replay import ReplayMemory, Sample
from collections import namedtuple
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import logging
from multiprocessing import Condition

TrainingSample = namedtuple('TrainingSample', ('state', 'policy', 'value', 'action_taken', 'goal', 'R', 'temporary_difference'))


class TrainingThread(mp.Process):
    def __init__(self,
            id : int,
            optimizer,
            device,
            network : torch.nn.Module,
            scene_glb : str,
            saver,
            max_t,
            terminal_image):

        super(TrainingThread, self).__init__()

        # Initialize the environment
        self.env = None
        self.scene_glb = scene_glb
        self.saver = saver
        self.max_t = max_t
        self.local_backbone_network = SharedNetwork().to(device)
        self.id = id
        self.terminal_image = terminal_image

        self.master_network = network
        self.optimizer = optimizer
        self.device = device

    def _sync_network(self):
        self.policy_network.load_state_dict(self.master_network.state_dict())

    def _ensure_shared_grads(self):
        for param, shared_param in zip(self.policy_network.parameters(), self.master_network.parameters()):
            if shared_param.grad is not None:
                return 
            shared_param._grad = param.grad 
    
    def get_action_space_size(self):
        return len(self.env.actions)

    def _initialize_thread(self):
        # self.logger = logging.getLogger('agent')
        # self.logger.setLevel(logging.INFO)
        #self.init_args['h5_file_path'] = lambda scene: h5_file_path.replace('{scene}', scene)
        #self.env = THORDiscreteEnvironment(self.scene, **self.init_args)
        self.env = HabitatDiscreteEnvironment(self.scene_glb, terminal_image=self.terminal_image)
        self.gamma = 0.99
        self.grad_norm = 40.0
        entropy_beta = 0.01
        self.local_t = 0
        self.action_space_size = self.get_action_space_size()

        self.criterion = ActorCriticLoss(entropy_beta)
        self.policy_network = nn.Sequential(SharedNetwork(), SceneSpecificNetwork(self.get_action_space_size())).to(self.device)
        # Initialize the episode
        self._reset_episode()
        self._sync_network()

    def _reset_episode(self):
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_max_q = -np.inf
        self.env.reset()

    def _forward_explore(self):
        # Does the evaluation end naturally?
        is_terminal = False
        terminal_end = False

        results = { "policy":[], "value": []}
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}

        # Plays out one game to end or max_t
        for t in range(self.max_t):
            state = { 
                "current": self.env.render(),
                "goal": self.env.render_target(),
            }

            x_processed = torch.from_numpy(state["current"])
            
            goal_processed = torch.from_numpy(state["goal"])

            print("Calculating action")
            (policy, value) = self.policy_network((x_processed, goal_processed,))

            # Store raw network output to use in backprop
            results["policy"].append(policy)
            results["value"].append(value)

            with torch.no_grad():
                (_, action,) = policy.max(0)
                action = F.softmax(policy, dim=0).multinomial(1).item()
            
            policy = policy.data.numpy()
            value = value.data.numpy()
            
            # Makes the step in the environment
            print("Stepping Agent with, ", action)
            self.env.step(action)

            # Receives the game reward
            is_terminal = self.env.is_terminal

            # ad-hoc reward for navigation
            reward = 10.0 if is_terminal else -0.01

            # Max episode length
            if self.episode_length > 5e3: is_terminal = True

            # Update episode stats
            self.episode_length += 1
            self.episode_reward += reward
            self.episode_max_q = max(self.episode_max_q, np.max(value))

            # clip reward
            reward = np.clip(reward, -1, 1)

            # Increase local time
            self.local_t += 1

            rollout_path["state"].append(state)
            rollout_path["action"].append(action)
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(is_terminal)

            if is_terminal:
                # TODO: add logging
                print('playout finished')
                print('Episode Length: ', self.episode_length)
                print('Episode Reward: ', self.episode_reward)
                print('Episode max_q', self.episode_max_q)

                terminal_end = True
                self._reset_episode()
                break

        if terminal_end:
            return 0.0, results, rollout_path
        else:
            x_processed = torch.from_numpy(self.env.render())
            goal_processed = torch.from_numpy(self.env.render_target())

            (_, value) = self.policy_network((x_processed, goal_processed,))
            return value.data.item(), results, rollout_path
    
    def _optimize_path(self, playout_reward: float, results, rollout_path):
        policy_batch = []
        value_batch = []
        action_batch = []
        temporary_difference_batch = []
        playout_reward_batch = []

        for i in reversed(range(len(results["value"]))):
            reward = rollout_path["rewards"][i]
            value = results["value"][i]
            action = rollout_path["action"][i]

            playout_reward = reward + self.gamma * playout_reward
            temporary_difference = playout_reward - value.data.item()

            policy_batch.append(results['policy'][i])
            value_batch.append(results['value'][i])
            action_batch.append(action)
            temporary_difference_batch.append(temporary_difference)
            playout_reward_batch.append(playout_reward)
        
        policy_batch = torch.stack(policy_batch, 0)
        value_batch = torch.stack(value_batch, 0)
        action_batch = torch.from_numpy(np.array(action_batch, dtype=np.int64))
        temporary_difference_batch = torch.from_numpy(np.array(temporary_difference_batch, dtype=np.float32))
        playout_reward_batch = torch.from_numpy(np.array(playout_reward_batch, dtype=np.float32))
        
        # Compute loss
        loss = self.criterion.forward(policy_batch, value_batch, action_batch, temporary_difference_batch, playout_reward_batch)
        loss = loss.sum()

        loss_value = loss.detach().numpy()
        self.optimizer.optimize(loss, 
            self.policy_network.parameters(), 
            self.master_network.parameters())

    def run(self, master = None):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("Thread ", self.id, ' ready')

        self._initialize_thread()
                
        if not master is None:
            print('Master Thread ', self.id, " started")
        else:
            print('Thread ', self.id, " started")

        try:
            self.env.reset()
            print("Env resetted")
            while True:
                print("While Loop started")
                self._sync_network()
                # Plays some samples
                playout_reward, results, rollout_path = self._forward_explore()
                # Train on collected samples
                self._optimize_path(playout_reward, results, rollout_path)
                
                print("Step Finished", self.optimizer.get_global_step())

                # Trigger save or other
                self.saver.after_optimization()                
                pass
        except Exception as e:
            print(e)
            # TODO: add logging
            #self.logger.error(e.msg)
            raise e