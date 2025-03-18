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
from utils.tool import get_offline_dataset, get_configs
from utils.dataset_env import make_replay_buffer_env
from utils.replay_buffer import ReplayBuffer
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from rich import print
from copy import deepcopy
from collections import deque
import gym
    
class Trainer():
    def __init__(self, config):        
        self.logger = SummaryWriter(config['exp_tag'])
        self.logger.add_text(
            "config",
            "|parametrix|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
        )
        self.config = config
        _, self.env = make_replay_buffer_env(config['dataset_name'])
        self.eval_env = deepcopy(self.env)

        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_high = float(self.env.action_space.high[0])
        action_low = float(self.env.action_space.low[0])
        self.replay_buffer = ReplayBuffer(buffer_size=config['buffer_size'], observation_dim=observation_dim, action_dim=action_dim)

        self.actor = Actor(
            latent_dim=observation_dim, 
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low).to(config['device'])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['lr_actor'])

        self.critic_1 = Critic(
            latent_dim=observation_dim,
            action_dim=action_dim).to(config['device'])
        self.target_1 = Critic(
            latent_dim=observation_dim,
            action_dim=action_dim).to(config['device'])
        self.target_1.load_state_dict(self.critic_1.state_dict())
        self.target_1.eval()
        self.critic_2 = Critic(
            latent_dim=observation_dim,
            action_dim=action_dim).to(config['device'])
        self.target_2 = Critic(
            latent_dim=observation_dim,
            action_dim=action_dim).to(config['device'])
        self.target_2.load_state_dict(self.critic_2.state_dict())
        self.target_2.eval()
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=config['lr_critic'])

        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(config['device'])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=config['device'])
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config['lr_alpha'])


        self.log_metric = {}

    def train(self):
        obs = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.config['device'])

        for self.global_step in trange(1, self.config['total_step']+1):
            if self.global_step < self.config['learn_start']:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _, _ = self.actor.get_action(obs)
                    action = action.squeeze().cpu().numpy()
            
            next_obs, reward, done, info = self.env.step(action)
            next_obs = torch.FloatTensor(next_obs).to(self.config['device'])


            self.replay_buffer.store(obs, action, reward, next_obs, done)

            obs = next_obs
            if done:
                obs = self.env.reset()
                obs = torch.FloatTensor(obs).to(self.config['device'])
            
            if self.global_step > self.config['learn_start']:
                b_obs, b_action, b_reward, b_next_obs, b_done = self.replay_buffer.sample(batch_size=self.config['batch_size'], device=self.config['device'])
                # critic
                with torch.no_grad():
                    p_next_action, p_next_log_pi, _ = self.actor.get_action(b_next_obs)
                    target_1_next_val = self.target_1(b_next_obs, p_next_action)
                    target_2_next_val = self.target_2(b_next_obs, p_next_action)
                    target_next_val = torch.min(target_1_next_val, target_2_next_val) - self.alpha * p_next_log_pi
                    next_q_val = b_reward + (1 - b_done) * self.config['gamma'] * (target_next_val)
                critic_1_val = self.critic_1(b_obs, b_action)
                critic_2_val = self.critic_2(b_obs, b_action)
                loss_critic_1 = F.mse_loss(critic_1_val, next_q_val)
                loss_critic_2 = F.mse_loss(critic_2_val, next_q_val)
                loss_critic = loss_critic_1 + loss_critic_2
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.critic_optimizer.step()
                # self.log_metric['loss_critic'] = loss_critic.item()

                # actor
                p_action, p_log_prob, _ = self.actor.get_action(b_obs)
                critic_1_q_val = self.critic_1(b_obs, p_action)
                critic_2_q_val = self.critic_2(b_obs, p_action)
                min_critic_q_val = torch.min(critic_1_q_val, critic_2_q_val)
                loss_actor = ((self.alpha * p_log_prob) - min_critic_q_val).mean()
                self.actor_optimizer.zero_grad()
                loss_actor.backward()
                self.actor_optimizer.step()
                # self.log_metric['loss_actor'] = loss_actor.item()

                # alpha
                with torch.no_grad():
                    _, p_log_prob, _ = self.actor.get_action(b_obs)
                loss_alpha = (-self.log_alpha.exp() * (p_log_prob + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
                # self.log_metric['loss_alpha'] = loss_alpha.item()

                for param, target_param in zip(self.critic_1.parameters(), self.target_1.parameters()):
                    target_param.data.copy_(self.config['soft_update_factor'] * param.data + (1 - self.config['soft_update_factor']) * target_param.data)
                for param, target_param in zip(self.critic_2.parameters(), self.target_2.parameters()):
                    target_param.data.copy_(self.config['soft_update_factor'] * param.data + (1 - self.config['soft_update_factor']) * target_param.data)

            if self.global_step % self.config['evaluate_freq'] == 0:
                re_mean, re_std = self.evaluate()                
                torch.save({
                    'step': self.global_step,
                    'actor': self.actor.state_dict(), 
                    'critic_1': self.critic_1.state_dict(), 
                    'critic_2': self.critic_2.state_dict(), 
                    're_mean': re_mean,
                    're_std': re_std,
                    },
                    f"new_checkpoints/delay_free_sac/{self.config['dataset_name']}_{self.config['seed']}.pth")


            self.logging()
        self.logger.close()
    


    def evaluate(self):
        self.log_metric['eval_r'] = []
        self.log_metric['eval_l'] = []
        obs = self.eval_env.reset()
        obs = torch.FloatTensor(obs).to(self.config['device'])


        while len(self.log_metric['eval_r']) != 10:
            with torch.no_grad():
                action, _, _ = self.actor.get_action(obs)
                action = action.squeeze().cpu().numpy()
            
            next_obs, reward, done, info = self.eval_env.step(action)
            next_obs = torch.FloatTensor(next_obs).to(self.config['device'])

            obs = next_obs
            if done:
                self.log_metric['eval_r'].append(info['episode']['r'])
                self.log_metric['eval_l'].append(info['episode']['l'])
                obs = self.eval_env.reset()
                obs = torch.FloatTensor(obs).to(self.config['device'])

        re_mean = np.mean(self.log_metric['eval_r'])
        re_std = np.std(self.log_metric['eval_r'])

        self.log_metric['eval_r'] = np.mean(self.log_metric['eval_r'])
        self.log_metric['eval_l'] = np.mean(self.log_metric['eval_l'])
        return re_mean, re_std



    def logging(self):
        for k in self.log_metric.keys():
            self.logger.add_scalar(k, self.log_metric[k], global_step=self.global_step)
        self.log_metric = {}
        

        


if __name__ == "__main__":
    configs = {
        "dataset_name": get_offline_dataset(policy='random'),
        "device":  ["cuda" if torch.cuda.is_available() else "cpu"],
        "seed": list(range(5)),  # Sets Gym, PyTorch and Numpy seeds
        "gamma": [0.99],
        "total_step": [int(1e6)],  # Max time steps to run environment
        "buffer_size": [int(1e6)],  # Replay buffer size
        "batch_size": [256],  # Batch size for all networks
        "lr_actor": [3e-4],
        "lr_critic": [1e-3],
        "lr_alpha": [1e-3],
        "soft_update_factor": [5e-3],
        "learn_start": [int(5e3)],
        "evaluate_freq": [int(1e4)],
    }
    configs = get_configs(configs)
    for config in configs:
        config['exp_tag'] = f"./logs/delay_free_sac/{config['dataset_name']}/SEED_{config['seed']}"
        if os.path.exists(config['exp_tag']):
            continue
        else:
            print(config)
        trainer = Trainer(config)
        trainer.train()
    