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

    def train_auto_encoder(self):
        auto_encoder = AutoEncoder(
            input_dim=self.observation_dim, 
            hidden_dim=256, 
            latent_dim=self.config['latent_dim']).to("cuda")
        auto_encoder_optimizer = optim.Adam(
            auto_encoder.parameters(), 
            lr=self.config['lr'])

        # load replay_buffer
        self.replay_buffer = ReplayBuffer(
            self.observation_dim, 
            action_dim=self.action_dim,
        )
        for policy in ['random', 'medium', 'expert']:
            dataset_name = f"{self.config['env_name']}-{policy}-v2"
            self.replay_buffer.load_d4rl_dataset(dataset_name)
        self.replay_buffer.normalize()

        for self.global_step in trange(500):
            sample_prior = self.replay_buffer.generate_sample_prior(batch_size=self.config['batch_size'])
            for indices in tqdm(sample_prior):
                observations, _, _, _, _ = self.replay_buffer.sample(indices)
                observations = observations.to("cuda")
                rec_observations, _ = auto_encoder(observations)
                loss = F.mse_loss(observations, rec_observations)
                auto_encoder_optimizer.zero_grad()
                loss.backward()
                auto_encoder_optimizer.step()
        torch.save({
            'step': self.global_step,
            'auto_encoder': auto_encoder.state_dict(),}, 
            f"new_checkpoints/auto_encoder/{self.config['env_name']}.pth")

                    


    def logging(self):
        for k in self.log_dict.keys():
            self.logger.add_scalar(k, self.log_dict[k], global_step=self.global_step)
        self.log_dict = {}

if __name__ == "__main__":
    configs = {
        "env_name": ["ant", "halfcheetah", "hopper", "walker2d"],
        "device":  ["cuda"],
        "lr": [1e-4],
        "delay": [8],
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

        config['exp_tag'] = f"new_logs/auto_encoder/{config['env_name']}"
        if os.path.exists(config['exp_tag']):
            continue
        trainer = Trainer(config)
        trainer.train_auto_encoder()


