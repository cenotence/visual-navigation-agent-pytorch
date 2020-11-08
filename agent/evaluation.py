

from agent.network import SharedNetwork, SceneSpecificNetwork
from agent.environment import HabitatDiscreteEnvironment
from agent.training import TrainingSaver, TOTAL_PROCESSED_FRAMES
from agent.utils import find_restore_point
import torch.nn.functional as F
import torch
import pickle
import os
import numpy as np
import re
from itertools import groupby

bench_dir = "agent/environment/bench/"

TASK_LIST = { bench_dir + 'Ballou.glb': bench_dir + 'Ballou_des.png' }
#              bench_dir + 'Roane.glb': bench_dir + 'Roane_des.png'
#            }

#GOAL_POSES = {'Ballou' : }
ACTION_SPACE_SIZE = 3
NUM_EVAL_EPISODES = 100
VERBOSE = False

def export_to_csv(data, file):
    import csv
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for k, g in groupby(sorted(data, key = lambda x: x[0]), key = lambda x: x[0]):
            g = list(g)
            header = [k, '']
            header.extend((np.mean(a) for a in list(zip(*g))[2:]))
            writer.writerow(header)
            for item in g:
                writer.writerow(list(item))
    print(f'CSV file stored "{file}"')

class Evaluation:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', torch.device('cuda:0'))
        self.shared_net = SharedNetwork().to(self.device)
        self.scene_nets = { key:SceneSpecificNetwork(ACTION_SPACE_SIZE).to(self.device) for key in TASK_LIST.keys() }

    @staticmethod
    def load_checkpoint(config, fail = True):
        checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        
        (base_name, restore_point) = find_restore_point(checkpoint_path, fail)
        print(f'Restoring from checkpoint {restore_point}')
        state = torch.load(open(os.path.join(os.path.dirname(checkpoint_path), base_name), 'rb'))
        evaluation = Evaluation(config)
        saver = TrainingSaver(evaluation.shared_net, evaluation.scene_nets, None, evaluation.config)
        print('Configuration')
        saver.restore(state)
        saver.print_config(offset = 4)            
        return evaluation

    def build_agent(self, scene_name):
        parent = self
        net = torch.nn.Sequential(parent.shared_net, parent.scene_nets[scene_name])
        class Agent:
            def __init__(self, initial_state, target):
                self.env = HabitatDiscreteEnvironment(self.scene_glb)
                self.env.reset()
                self.net = net

            @staticmethod
            def get_parameters():
                return net.parameters()

            def act(self):
                with torch.no_grad():
                    state = torch.Tensor(self.env.render()).to(parent.device)
                    target = torch.Tensor(self.env.render_target()).to(parent.device)
                    (policy, value,) = net.forward((state, target,))
                    action = F.softmax(policy, dim=0).multinomial(1).cpu().data.numpy()[0]

                self.env.step(action)
                return (self.env.is_terminal, self.env.collided, self.env.reward)
        return Agent
        
    
    def run(self):
        scene_stats = dict()
        resultData = []

        for scene_scope, image in TASK_LIST.items():
            scene_net = self.scene_nets[scene_scope]
            scene_stats[scene_scope] = list()
            env = HabitatDiscreteEnvironment(
                scene_scope, 
                terminal_image=image
            )
            ep_rewards = []
            ep_lengths = []
            ep_collisions = []
            ep_normalized_lengths = []

            env.reset()
            terminal = False
            ep_reward = 0
            ep_collision = 0
            ep_t = 0

            while not terminal:
                state = torch.Tensor(env.render()).permute(0,3,1,2).to(self.device)
                
                target = torch.Tensor(env.render_target()).permute(0,3,1,2).to(self.device)
                
                (policy, value,) = scene_net.forward(self.shared_net.forward((state, target,)))

                with torch.no_grad():
                    action = F.softmax(policy, dim=0).multinomial(1).cpu().data.numpy()[0]
                print("Applied action: ", action)
                env.step(action)
                terminal = env.is_terminal

                ep_reward += env.reward
                ep_t += 1       

            ep_lengths.append(ep_t)
            ep_rewards.append(ep_reward)
            ep_collisions.append(ep_collision)
            ep_normalized_lengths.append(ep_t)
            if VERBOSE: print("episode #{} ends after {} steps".format(i_episode, ep_t))

            print('evaluation: %s' % (scene_scope))
            print('mean episode reward: %.2f' % np.mean(ep_rewards))
            print('mean episode length: %.2f' % np.mean(ep_lengths))
            print('mean episode collision: %.2f' % np.mean(ep_collisions))
            print('mean normalized episode length: %.2f' % np.mean(ep_normalized_lengths))
            scene_stats[scene_scope].extend(ep_lengths)
            resultData.append((scene_scope, np.mean(ep_rewards), np.mean(ep_lengths), np.mean(ep_collisions), np.mean(ep_normalized_lengths),))
            print('\nResults (average trajectory length):')
        
        for scene_scope in scene_stats:
            print('%s: %.2f steps'%(scene_scope, np.mean(scene_stats[scene_scope])))
        
        if 'csv_file' in self.config and self.config['csv_file'] is not None:
            export_to_csv(resultData, self.config['csv_file'])
'''
        for scene_scope, items in TASK_LIST.items():
            if len(self.config['test_scenes']) != 0 and not scene_scope in self.config['test_scenes']:
                continue

            for task_scope in items:
                env = HabitatDiscreteEnvironment(
                    list(TASK_LIST.keys())[0], 
                    terminal_image=TASK_LIST[list(TASK_LIST.keys())[0]]
                )

                ep_rewards = []
                ep_lengths = []
                ep_collisions = []
                ep_normalized_lengths = []
                for (i_episode, start) in enumerate(env.get_initial_states(int(task_scope))):
                    env.reset()
                    terminal = False
                    ep_reward = 0
                    ep_collision = 0
                    ep_t = 0
                    hitting_time = hitting_times[start, int(task_scope)]
                    shortest_path = shortest_paths[start, int(task_scope)]

                    while not terminal:
                        state = torch.Tensor(env.render())
                        target = torch.Tensor(env.render_target())
                        (policy, value,) = scene_net.forward(self.shared_net.forward((state, target,)))

                        with torch.no_grad():
                            action = F.softmax(policy, dim=0).multinomial(1).data.numpy()[0]
                        env.step(action)
                        terminal = env.is_terminal

                        if ep_t == hitting_time: break
                        if env.collided: ep_collision += 1
                        ep_reward += env.reward
                        ep_t += 1                   


                    ep_lengths.append(ep_t)
                    ep_rewards.append(ep_reward)
                    ep_collisions.append(ep_collision)
                    ep_normalized_lengths.append(min(ep_t, hitting_time) / shortest_path)
                    if VERBOSE: print("episode #{} ends after {} steps".format(i_episode, ep_t))

                    
                print('evaluation: %s %s' % (scene_scope, task_scope))
                print('mean episode reward: %.2f' % np.mean(ep_rewards))
                print('mean episode length: %.2f' % np.mean(ep_lengths))
                print('mean episode collision: %.2f' % np.mean(ep_collisions))
                print('mean normalized episode length: %.2f' % np.mean(ep_normalized_lengths))
                scene_stats[scene_scope].extend(ep_lengths)
                resultData.append((scene_scope, str(task_scope), np.mean(ep_rewards), np.mean(ep_lengths), np.mean(ep_collisions), np.mean(ep_normalized_lengths),))

        print('\nResults (average trajectory length):')
        for scene_scope in scene_stats:
            print('%s: %.2f steps'%(scene_scope, np.mean(scene_stats[scene_scope])))
        
        if 'csv_file' in self.config and self.config['csv_file'] is not None:
            export_to_csv(resultData, self.config['csv_file'])
'''
'''
# Load weights trained on tensorflow
data = pickle.load(open(os.path.join(__file__, '..\\..\\weights.p'), 'rb'), encoding='latin1')
def convertToStateDict(data):
    return {key:torch.Tensor(v) for (key, v) in data.items()}

shared_net.load_state_dict(convertToStateDict(data['navigation']))
for key in TASK_LIST.keys():
    scene_nets[key].load_state_dict(convertToStateDict(data[f'navigation/{key}']))'''
