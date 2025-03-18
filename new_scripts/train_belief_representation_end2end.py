import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.network import AutoEncoder, GRU_Dynamic, MLP_Dynamic, TRANS_Dynamic
from utils.tool import get_configs, get_auto_encoder
from utils.dataset import ReplayBuffer, DelayBuffer
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
import gym
import numpy as np

class Trainer():
    def __init__(self, config):
        self.config = config
        self.logger = SummaryWriter(config['exp_tag'])
        self.logger.add_text(
            "config",
            "|parametrix|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
        )
        self.log_dict = {}


        env = gym.make(f"{config['env_name']}-random-v2")
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]


    def train_latent_dynamic(self):
        # load replay_buffer
        self.replay_buffer = DelayBuffer(
            self.observation_dim, 
            action_dim=self.action_dim,
            delay=self.config['delay'],
        )
        for policy in ['random', 'medium', 'expert']:
        # for policy in ['random']:
            dataset_name = f"{config['env_name']}-{policy}-v2"
            self.replay_buffer.load_d4rl_dataset(dataset_name)
        self.replay_buffer.normalize_reward()

        self.global_step = 0
        self.auto_encoder = AutoEncoder(
            input_dim=self.observation_dim, 
            hidden_dim=256, 
            latent_dim=self.config['latent_dim']).to("cuda")
        

        if self.config['dynamic_type'] == 'mlp':
            self.dynamic = MLP_Dynamic(latent_dim=self.config['latent_dim'], 
                                       condition_dim=self.action_dim, 
                                       hidden_dim=self.config['latent_dim']).to(config['device'])
        elif self.config['dynamic_type'] == 'gru':
            self.dynamic = GRU_Dynamic(latent_dim=self.config['latent_dim'], 
                                       condition_dim=self.action_dim, 
                                       hidden_dim=self.config['latent_dim']).to(config['device'])
        elif self.config['dynamic_type'] == 'trans':
            self.dynamic = TRANS_Dynamic(latent_dim=config['latent_dim'], 
                                         condition_dim=self.action_dim, 
                                         seq_len=self.config['delay'], 
                                         hidden_dim=self.config['latent_dim'],
                                         num_layers=self.config['num_layers'],
                                         num_heads=self.config['num_heads'],
                                         ).to(self.config['device'])
        else:
            raise NotImplementedError

        self.optimizer = optim.Adam(list(self.auto_encoder.parameters()) + list(self.dynamic.parameters()), lr=self.config['lr'])

        for self.global_step in trange(100):
            self.replay_buffer.generate_sample_prior()
            if self.config['dynamic_type'] == 'mlp':
                self.train_mlp_dynamic()
            elif self.config['dynamic_type'] == 'gru':
                self.train_gru_dynamic()
            elif self.config['dynamic_type'] == 'trans':
                self.train_trans_dynamic()
            else:
                raise NotImplementedError
            torch.save({
                'step': self.global_step,
                'auto_encoder': self.auto_encoder.state_dict(), 
                'latent_dynamic': self.dynamic.state_dict(), 
                'reward_mean': self.replay_buffer.reward_mean,
                'reward_std': self.replay_buffer.reward_std,
                },
                f"new_checkpoints/trans_reward/{self.config['env_name']}_{self.config['dynamic_type']}_Delay_{self.config['delay']}.pth")

        

    def train_mlp_dynamic(self):
        for indices in tqdm(self.replay_buffer._sample_prior):
            states, actions, rewards, dones, masks = self.replay_buffer.sample(indices)
            states = states.to(self.config['device'])
            actions = actions.to(self.config['device'])
            masks = masks[:, 1:, 0].to(self.config['device'])

            with torch.no_grad():
                _, latents = self.auto_encoder(states)

            z = self.dynamic(latents[:, :-1, :], actions[:, :-1, :])
            loss = F.mse_loss(z, latents[:, 1:, :], reduction='none').mean(-1)
            loss = ((1-masks) * loss).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.log_dict["loss"] = loss.item()
    
    def train_gru_dynamic(self):
        for indices in tqdm(self.replay_buffer._sample_prior):
            states, actions, rewards, dones, masks = self.replay_buffer.sample(indices)
            states = states.to(self.config['device'])
            actions = actions.to(self.config['device'])
            masks = masks[:, 1:, 0].to(self.config['device'])

            with torch.no_grad():
                _, latents = self.auto_encoder(states)
                h = self.dynamic.init_hidden(states.shape[0]).to(self.config['device'])

            z_target = latents[:, 1:, :]
            z, h = self.dynamic(latents[:, :-1, :], actions[:, :-1, :], h)
            loss = F.mse_loss(z, z_target)
            loss = F.mse_loss(z, latents[:, 1:], reduction='none').mean(-1)
            loss = ((1-masks) * loss).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.log_dict["loss"] = loss.item()

    def train_trans_dynamic(self):
        for indices in tqdm(self.replay_buffer._sample_prior):
            states, actions, rewards, dones, masks = self.replay_buffer.sample(indices)
            states = states.to(self.config['device'])
            actions = actions.to(self.config['device'])
            rewards = rewards.to(self.config['device'])
            masks = masks[:, 1:, 0].to(self.config['device'])

            latents = self.auto_encoder.encode(states)
            timesteps = torch.arange(0, self.config['delay'], dtype=torch.int32).to(self.config['device'])
            z = self.dynamic(latents=latents[:, :1, :], 
                             actions=actions[:, :self.config['delay'], :],
                             rewards=rewards[:, :self.config['delay'], :],
                             timesteps=timesteps,
                             masks=masks)
            rec_states = self.auto_encoder.decode(z)
            loss = F.mse_loss(rec_states, states[:, 1:, :], reduction='none').mean(-1)
            loss = ((1-masks) * loss).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.log_dict["loss"] = loss.item()


    def logging(self):
        for k in self.log_dict.keys():
            self.logger.add_scalar(k, self.log_dict[k], global_step=self.global_step)
        self.log_dict = {}

if __name__ == "__main__":
    configs = {
        "env_name": ["halfcheetah", "hopper", "walker2d"],
        "device":  ["cuda"],
        "lr": [1e-4],
        "delay": [8, 16, 32, 64, 128],
        "batch_size": [256],
        "latent_dim": [256],
        "hidden_dim": [256],
        "num_layers": [10],
        "num_heads": [4],
        "attention_dropout": [0.1],
        "residual_dropout": [0.1],
        "hidden_dropout": [0.1],
        "dynamic_type": ['trans'],
    }
    configs = get_configs(configs)
    for config in configs:

        config['exp_tag'] = f"new_logs_end2end_reward/{config['dynamic_type']}/{config['env_name']}/{config['delay']}"
        if os.path.exists(config['exp_tag']):
            continue
        trainer = Trainer(config)
        trainer.train_latent_dynamic()