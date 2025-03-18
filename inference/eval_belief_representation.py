import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.network import Actor, Critic
from utils.network import AutoEncoder, GRU_Dynamic, MLP_Dynamic, TRANS_Dynamic
from utils.tool import get_offline_dataset, get_auto_encoder, get_latent_dynamic, get_configs
from utils.dataset_env import make_replay_buffer_env
from utils.replay_buffer import LatentBuffer
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from rich import print
from copy import deepcopy
from collections import deque
import gym
    
class Trainer():
    def __init__(self, config):
        self.config = config
        self.env = gym.make(f"{config['env_name']}-random-v2")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_high = float(self.env.action_space.high[0])
        action_low = float(self.env.action_space.low[0])

        self.auto_encoder = AutoEncoder(
            input_dim=observation_dim, 
            hidden_dim=256, 
            latent_dim=config['latent_dim']).to(config['device'])
        checkpoint = torch.load(f"new_checkpoints/auto_encoder/{config['env_name']}.pth", map_location=torch.device('cpu'))
        self.auto_encoder.load_state_dict(checkpoint['auto_encoder'])
        self.auto_encoder.eval()
        print('loaded auto encoder')



        if config['dynamic_type'] == 'mlp':
            self.latent_dynamic = MLP_Dynamic(latent_dim=config['latent_dim'], 
                                       condition_dim=action_dim, 
                                       hidden_dim=config['latent_dim']).to(config['device'])
            checkpoint = torch.load(f"new_checkpoints/latent_dynamic/{config['env_name']}_{config['dynamic_type']}_Delay_128.pth", map_location=torch.device('cpu'))
        elif config['dynamic_type'] == 'gru':
            self.latent_dynamic = GRU_Dynamic(latent_dim=config['latent_dim'], 
                                       condition_dim=action_dim, 
                                       hidden_dim=config['latent_dim']).to(config['device'])
            checkpoint = torch.load(f"new_checkpoints/latent_dynamic/{config['env_name']}_{config['dynamic_type']}_Delay_128.pth", map_location=torch.device('cpu'))
        elif config['dynamic_type'] == 'trans':
            self.latent_dynamic = TRANS_Dynamic(latent_dim=config['latent_dim'], 
                                         condition_dim=action_dim, 
                                         seq_len=128, 
                                         hidden_dim=config['latent_dim'],
                                         num_layers=config['num_layers'],
                                         num_heads=config['num_heads'],
                                         ).to(config['device'])
            # checkpoint = torch.load(f"new_checkpoints/latent_dynamic/{config['env_name']}_{config['dynamic_type']}_Delay_128.pth", map_location=torch.device('cpu'))
            checkpoint = torch.load(f"new_checkpoints/latent_dynamic/{config['env_name']}_{config['dynamic_type']}_Delay_{config['delay']}.pth", map_location=torch.device('cpu'))
        else:
            raise NotImplementedError

        checkpoint = torch.load(f"new_checkpoints/latent_dynamic/{config['env_name']}_{config['dynamic_type']}_Delay_128.pth", map_location=torch.device('cpu'))
        self.latent_dynamic.load_state_dict(checkpoint['latent_dynamic'])
        self.latent_dynamic.eval()
        print('loaded latent dynamic')

        self.actor = Actor(
            latent_dim=observation_dim, 
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low).to(config['device'])
        checkpoint = torch.load(f"new_checkpoints/delay_free_sac/{config['env_name']}-random-v2_{config['seed']}.pth", map_location=torch.device('cpu'))
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor.eval()
        print(checkpoint['step'])
        print('loaded delay free actor')
    
    def get_next_latent_mlp_dy(self, next_latent, actions):
        next_latent = next_latent.unsqueeze(0)
        for i in range(len(actions)):
            next_latent = self.latent_dynamic(
                next_latent, 
                torch.FloatTensor(actions[i]).unsqueeze(0).to(self.config['device'])
            )
        next_latent = next_latent.squeeze(0)
        return next_latent

    def get_next_latent_gru_dy(self, next_latent, actions):
        next_latent = next_latent.unsqueeze(0)
        h = self.latent_dynamic.init_hidden(1).to(self.config['device']).squeeze(1)
        for i in range(len(actions)):
            next_latent, h = self.latent_dynamic(
                next_latent, 
                torch.FloatTensor(actions[i]).unsqueeze(0).to(self.config['device']),
                h
            )
        next_latent = next_latent.squeeze(0)
        return next_latent

    def get_next_latent_trans_dy(self, next_latent, actions):
        delayed_idx = len(actions) - 1
        next_latent = next_latent.unsqueeze(0)
        timesteps = torch.arange(0, 128, dtype=torch.int32).to(self.config['device'])
        masks = torch.zeros(len(actions)).unsqueeze(0).to(self.config['device'])
        pad_masks = torch.ones(128 - len(actions)).unsqueeze(0).to(self.config['device'])
        masks = torch.concat((masks, pad_masks), dim=-1)
        # print(masks.shape)
        action_dim = actions[0].shape[0]
        pad_actions = torch.zeros((1, 128 - len(actions), action_dim)).to(self.config['device'])
        # print(pad_actions.shape)
        # print(torch.FloatTensor(np.array(list(actions))).unsqueeze(0).to(self.config['device']).shape)
        # exit()
        next_latent = self.latent_dynamic(
            latents=next_latent, 
            actions=torch.concat((torch.FloatTensor(np.array(list(actions))).unsqueeze(0).to(self.config['device']), pad_actions), dim=1),
            timesteps=timesteps,
            masks=masks
        )
        next_latent = next_latent[:, delayed_idx, :].squeeze(0)
        return next_latent

    def evaluate(self):
        eval_re = []
        eval_len = []

        obs = self.env.reset()
        with torch.no_grad():
            latent = self.auto_encoder.encode(torch.FloatTensor(obs).to(self.config['device']))
            rec_obs = self.auto_encoder.decode(latent)
        delayed_deque = {
            'obs': deque(maxlen=self.config['delay'] + 1),
            'action': deque(maxlen=self.config['delay']),
            'reward': deque(maxlen=self.config['delay']),
            'done': deque(maxlen=self.config['delay']),
        }
        delayed_deque['obs'].append(torch.FloatTensor(obs).to(self.config['device']))

        while len(eval_re) < 10:
            with torch.no_grad():
                action, _, _ = self.actor.get_action(rec_obs)
                action = action.squeeze().cpu().numpy()
            
            next_obs, reward, done, info = self.env.step(action)

            delayed_deque['obs'].append(torch.FloatTensor(next_obs).to(self.config['device']))
            delayed_deque['action'].append(action)
            delayed_deque['reward'].append(reward)
            delayed_deque['done'].append(done)

            # estimate next latent
            with torch.no_grad():
                next_latent = self.auto_encoder.encode(delayed_deque['obs'][0])
                if self.config['dynamic_type'] == 'mlp':
                    # mlp
                    next_latent = self.get_next_latent_mlp_dy(next_latent, delayed_deque['action'])
                elif self.config['dynamic_type'] == 'gru':
                    # gru
                    next_latent = self.get_next_latent_gru_dy(next_latent, delayed_deque['action'])
                elif self.config['dynamic_type'] == 'trans':
                    # trans
                    next_latent = self.get_next_latent_trans_dy(next_latent, delayed_deque['action'])                
                else:
                    raise NotImplementedError
                next_rec_obs = self.auto_encoder.decode(next_latent)

            latent = next_latent
            obs = next_obs
            rec_obs = next_rec_obs
            if done:
                eval_re.append(info['episode']['r'])
                eval_len.append(info['episode']['l'])
                obs = self.env.reset()
                with torch.no_grad():
                    latent = self.auto_encoder.encode(torch.FloatTensor(obs).to(self.config['device']))
                    rec_obs = self.auto_encoder.decode(latent)
                delayed_deque = {
                    'obs': deque(maxlen=self.config['delay'] + 1),
                    'action': deque(maxlen=self.config['delay']),
                    'reward': deque(maxlen=self.config['delay']),
                    'done': deque(maxlen=self.config['delay']),
                }
                delayed_deque['obs'].append(torch.FloatTensor(obs).to(self.config['device']))


        
        eval_re = np.mean(eval_re)
        eval_len = np.mean(eval_len)
        return eval_re, eval_len

        


if __name__ == "__main__":
    configs = {
        "env_name": ["halfcheetah", "hopper", "walker2d"],
        "device":  ["cuda" if torch.cuda.is_available() else "cpu"],
        "seed": list(range(5)),  # Sets Gym, PyTorch and Numpy seeds
        "gamma": [0.99],
        "latent_dim": [256],
        "num_layers": [10],
        "num_heads": [4],
        "attention_dropout": [0.1],
        "residual_dropout": [0.1],
        "hidden_dropout": [0.1],
        "soft_update_factor": [5e-3],
        "learn_start": [int(5e3)],
        "evaluate_freq": [int(1e4)],
        "delay": [int(8), int(16), int(32), int(64), int(128)],
        "dynamic_type": ['mlp', 'gru', 'trans'],
    }
    configs = get_configs(configs)
    for config in configs:

        with open('inference.txt', 'r') as file:
            lines = file.readlines()
            has_ran = False
            for line in lines:
                if f"{config['env_name']}, {config['delay']}, {config['dynamic_type']}, {config['seed']}" in line:
                    has_ran = True
                    break
        if has_ran:
            continue

        trainer = Trainer(config)
        eval_re, eval_len = trainer.evaluate()
        with open('inference.txt', 'a') as file:
            file.write(f"{config['env_name']}, {config['delay']}, {config['dynamic_type']}, {config['seed']}, {eval_re}, {eval_len} \n")
        # print(eval_re, eval_len)
    