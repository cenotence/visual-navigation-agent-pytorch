from agent.network import SharedNetwork, SceneSpecificNetwork
from agent.training_thread import TrainingThread
from agent.optim import SharedRMSprop
from typing import Collection, List
import torch
import torch.nn as nn
import torch.multiprocessing as mp 
import logging
import sys
import torch
import os
import threading
from contextlib import suppress
import re


TOTAL_PROCESSED_FRAMES = 10000

bench_dir = "agent/environment/bench/"
#Env Name, Image
TASK_LIST = { bench_dir + 'Ballou.glb': bench_dir + 'Ballou_des.png' }
#              bench_dir + 'Stokes.glb': bench_dir + 'Stokes_des.png' ,
#              bench_dir + 'Roane.glb': bench_dir + 'Roane_des.png'
#            }
#TASK_LIST = { bench_dir + 'Arkansaw.glb': bench_dir + 'Arkansaw_des.png' }
#           , bench_dir + 'Hillsdale.glb': bench_dir + 'Hillsdale_des.png' }

DEFAULT_CONFIG = {
    'saving_period': 10000,
    'checkpoint_path': 'model/checkpoint-{checkpoint}.pth',
    'grad_norm': 40.0,
    'gamma': 0.99,
    'entropy_beta': 0.01,
    'max_t': 1,
}


class TrainingSaver:
    def __init__(self, shared_network, scene_networks, optimizer, config):
        self.config = config
        n_config = DEFAULT_CONFIG.copy()
        n_config.update(config)
        self.config.update(n_config)
        self.checkpoint_path = self.config['checkpoint_path']
        self.saving_period = self.config['saving_period']
        self.shared_network = shared_network
        self.scene_networks = scene_networks
        self.optimizer = optimizer        

    def after_optimization(self):
        iteration = self.optimizer.get_global_step()
        if iteration % self.saving_period == 0:
            self.save()

    def print_config(self, offset: int = 0):
        for key, val in self.config.items():
            print(key, val)
            #print((" " * offset) + f"{key}: {val}")
        pass

    def save(self):
        iteration = self.optimizer.get_global_step()
        filename = self.checkpoint_path.replace('{checkpoint}', str(iteration))
        model = dict()
        model['navigation'] = self.shared_network.state_dict()
        for key, val in self.scene_networks.items():
            model['navigation/' + str(key)] = val.state_dict()
        model['optimizer'] = self.optimizer.state_dict()
        model['config'] = self.config
        
        with suppress(FileExistsError):
            os.makedirs(os.path.dirname(filename))

        torch.save(model, open(filename, 'wb'))

    def restore(self, state):
        if 'optimizer' in state and self.optimizer is not None: self.optimizer.load_state_dict(state['optimizer'])
        if 'config' in state: 
            n_config = state['config'].copy()
            n_config.update(self.config)
            self.config.update(n_config)

        self.shared_network.load_state_dict(state['navigation'])

        tasks = self.config.get('tasks', TASK_LIST)
        for scene in tasks.keys():
            self.scene_networks[scene].load_state_dict(state['navigation/'+str(scene)])


class TrainingOptimizer:
    def __init__(self, grad_norm, optimizer: torch.optim.Optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_norm = grad_norm
        self.global_step = torch.tensor(0)
        self.lock = mp.Lock()

    def state_dict(self):
        state_dict = dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict["global_step"] = self.global_step
        return state_dict

    def share_memory(self):
        self.global_step.share_memory_()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.global_step.copy_(state_dict['global_step'])
    
    def get_global_step(self):
        return self.global_step.item()

        
    def _ensure_shared_grads(self, local_params, shared_params):
        for param, shared_param in zip(local_params, shared_params):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def optimize(self, loss, local_params, shared_params):
        local_params = list(local_params)
        shared_params = list(shared_params)

        # Fix the optimizer property after unpickling
        self.scheduler.optimizer = self.optimizer
        self.scheduler.step(self.global_step.item())

        # Increment step
        with self.lock:
            self.global_step.copy_(torch.tensor(self.global_step.item() + 1))
            
        self.optimizer.zero_grad()

        # Calculate the new gradient with the respect to the local network
        loss.backward()

        # Clip gradient
        torch.nn.utils.clip_grad_norm_(local_params, self.grad_norm)
            
        self._ensure_shared_grads(local_params, shared_params)
        self.optimizer.step()


class AnnealingLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        super(AnnealingLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1.0 - self.last_epoch / self.total_epochs)
                for base_lr in self.base_lrs]


class Training:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.logger = self._init_logger()
        self.learning_rate = config.get('learning_rate')
        self.rmsp_alpha = config.get('rmsp_alpha')
        self.rmsp_epsilon = config.get('rmsp_epsilon')
        self.grad_norm = config.get('grad_norm', 40.0)
        self.tasks = config.get('tasks', TASK_LIST)
        self.checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        self.max_t = config.get('max_t', 5)
        self.total_epochs = TOTAL_PROCESSED_FRAMES // self.max_t
        self.initialize()
    
    def load_checkpoint(config, fail = True):
        device = torch.device('cpu')
        checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        max_t = config.get('max_t', 5)
        total_epochs = TOTAL_PROCESSED_FRAMES // max_t
        files = os.listdir(os.path.dirname(checkpoint_path))
        base_name = os.path.basename(checkpoint_path)
        
        # Find latest checkpoint
        # TODO: improve speed
        restore_point = None
        if base_name.find('{checkpoint}') != -1:
            regex = re.escape(base_name).replace(re.escape('{checkpoint}'), '(\d+)')
            points = [(fname, int(match.group(1))) for (fname, match) in ((fname, re.match(regex, fname),) for fname in files) if not match is None]
            if len(points) == 0:
                if fail:
                    raise Exception('Restore point not found')
                else: return None
            
            (base_name, restore_point) = max(points, key = lambda x: x[1])

            
        print('Restoring from checkpoint ', restore_point)
        state = torch.load(open(os.path.join(os.path.dirname(checkpoint_path), base_name), 'rb'))
        training = Training(device, state['config'] if 'config' in state else config)
        training.saver.restore(state) 
        print('Configuration')
        training.saver.print_config(offset = 4)       
        return training
    
    def initialize(self):
        self.shared_network = SharedNetwork().to(self.device)
        # 12 actions Roll pitch yaw surge sway heave
        self.scene_networks = { key:SceneSpecificNetwork(8).to(self.device) for key in TASK_LIST.keys() }

        self.shared_network.share_memory()
        for net in self.scene_networks.values():
            net.share_memory()
        
        parameters = list(self.shared_network.parameters())
        for net in self.scene_networks.values():
            parameters.extend(net.parameters())
        
        optimizer = SharedRMSprop(parameters, eps=self.rmsp_epsilon, alpha=self.rmsp_alpha, lr=self.learning_rate)
        optimizer.share_memory()

        scheduler = AnnealingLRScheduler(optimizer, self.total_epochs)

        optimizer_wrapper = TrainingOptimizer(self.grad_norm, optimizer, scheduler)
        self.optimizer = optimizer_wrapper
        optimizer_wrapper.share_memory()

        self.saver = TrainingSaver(self.shared_network, self.scene_networks, self.optimizer, self.config)
    
    def run(self):
        self.logger.info("Training started")

        # Prepare threads
        branches = [(scene, TASK_LIST.get(scene)) for scene in TASK_LIST.keys()]
        def _createThread(id, task):
            (scene_glb, target_img) = task
            net = nn.Sequential(self.shared_network, self.scene_networks[scene_glb]).to(self.device)
            net.share_memory()
            return TrainingThread(
                id = id,
                optimizer = self.optimizer,
                device = self.device,
                network = net,
                scene_glb = scene_glb,
                saver = self.saver,
                max_t = self.max_t,
                terminal_image = target_img)

        self.threads = [_createThread(i, task) for i, task in enumerate(branches)]
        
        try:
            for thread in self.threads:
                thread.start()

            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            # we will save the training
            print('Saving training session')
            self.saver.save()

    def _init_logger(self):
        logger = logging.getLogger('agent')
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        return logger

