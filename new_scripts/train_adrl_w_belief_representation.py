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
from utils.replay_buffer import LatentBuffer, LatentRecBuffer
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from rich import print
from copy import deepcopy
from collections import deque
import gym
import random
    
class Trainer():
    def __init__(self, config):
        self.logger = SummaryWriter(config['exp_tag'])
        self.logger.add_text(
            "config",
            "|parametrix|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
        )
        self.config = config
        self.env = gym.make(f"{config['env_name']}-random-v2")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.eval_env = deepcopy(self.env)
        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_high = float(self.env.action_space.high[0])
        action_low = float(self.env.action_space.low[0])

        self.auto_encoder = AutoEncoder(
            input_dim=observation_dim, 
            hidden_dim=256, 
            latent_dim=config['latent_dim']).to(config['device'])
        # checkpoint = torch.load(f"new_checkpoints/end2end/{config['env_name']}.pth", map_location=torch.device('cpu'))
        # checkpoint = torch.load(f"new_checkpoints/end2end/{config['env_name']}_{config['dynamic_type']}_Delay_{config['delay']}.pth", map_location=torch.device('cpu'))

        # checkpoint = torch.load(f"new_checkpoints/auto_encoder/{config['env_name']}.pth", map_location=torch.device('cpu'))
        checkpoint = torch.load(f"new_checkpoints/trans_reward/{config['env_name']}_trans_Delay_{config['delay']}.pth", map_location=torch.device('cpu'))

        self.auto_encoder.load_state_dict(checkpoint['auto_encoder'])
        self.auto_encoder.eval()
        print('loaded auto encoder')



        if config['dynamic_type'] == 'mlp':
            self.latent_dynamic = MLP_Dynamic(latent_dim=config['latent_dim'], 
                                       condition_dim=action_dim, 
                                       hidden_dim=config['latent_dim']).to(config['device'])
        elif config['dynamic_type'] == 'gru':
            self.latent_dynamic = GRU_Dynamic(latent_dim=config['latent_dim'], 
                                       condition_dim=action_dim, 
                                       hidden_dim=config['latent_dim']).to(config['device'])
        elif config['dynamic_type'] == 'trans':
            self.latent_dynamic = TRANS_Dynamic(latent_dim=config['latent_dim'], 
                                         condition_dim=action_dim, 
                                         seq_len=config['delay'], 
                                         hidden_dim=config['latent_dim'],
                                         num_layers=config['num_layers'],
                                         num_heads=config['num_heads'],
                                         ).to(config['device'])
        else:
            raise NotImplementedError

        # checkpoint = torch.load(f"new_checkpoints/latent_dynamic/{config['env_name']}_{config['dynamic_type']}_Delay_{config['delay']}.pth", map_location=torch.device('cpu'))
        self.latent_dynamic.load_state_dict(checkpoint['latent_dynamic'])
        self.latent_dynamic.eval()
        print('loaded latent dynamic')
        self.reward_mean = checkpoint['reward_mean']
        self.reward_std = checkpoint['reward_std']
        print(checkpoint['step'])
        


        self.replay_buffer = LatentRecBuffer(buffer_size=config['buffer_size'], observation_dim=observation_dim, latent_dim=config['latent_dim'], action_dim=action_dim)

        self.actor = Actor(
            # latent_dim=config['latent_dim'], 
            latent_dim=observation_dim, 
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low).to(config['device'])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['lr_actor'])

        self.df_actor = Actor(
            latent_dim=observation_dim, 
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low).to(config['device'])
        self.df_actor_optimizer = optim.Adam(self.df_actor.parameters(), lr=config['lr_actor'])

        self.critic_1 = Critic(latent_dim=observation_dim, action_dim=action_dim).to(config['device'])
        self.target_1 = Critic(latent_dim=observation_dim, action_dim=action_dim).to(config['device'])
        self.target_1.load_state_dict(self.critic_1.state_dict())
        self.target_1.eval()
        self.critic_2 = Critic(latent_dim=observation_dim, action_dim=action_dim).to(config['device'])
        self.target_2 = Critic(latent_dim=observation_dim, action_dim=action_dim).to(config['device'])
        self.target_2.load_state_dict(self.critic_2.state_dict())
        self.target_2.eval()
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=config['lr_critic'])

        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(config['device'])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=config['device'])
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config['lr_alpha'])

        self.log_df_alpha = torch.zeros(1, requires_grad=True, device=config['device'])
        self.df_alpha = self.log_df_alpha.exp().item()
        self.df_alpha_optimizer = optim.Adam([self.log_df_alpha], lr=config['lr_alpha'])

        self.log_metric = {}

    def train(self):
        obs = self.env.reset()
        with torch.no_grad():
            latent = self.auto_encoder.encode(torch.FloatTensor(obs).to(self.config['device']))
        delayed_deque = {
            'obs': deque(maxlen=self.config['delay'] + 1),
            'action': deque(maxlen=self.config['delay']),
            'reward': deque(maxlen=self.config['delay']),
        }
        delayed_deque['obs'].append(torch.FloatTensor(obs).to(self.config['device']))
        rec_obs = torch.FloatTensor(obs).to(self.config['device'])

        for self.global_step in trange(1, self.config['total_step']+1):
            if self.global_step < self.config['learn_start']:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    # action, _, _ = self.actor.get_action(latent)
                    action, _, _ = self.actor.get_action(rec_obs)
                    value_1 = self.target_1(delayed_deque['obs'][-1], action)
                    value_2 = self.target_2(delayed_deque['obs'][-1], action)
                    value = torch.min(value_1, value_2)

                    df_action, _, _ = self.df_actor.get_action(delayed_deque['obs'][-1])
                    df_value_1 = self.target_1(delayed_deque['obs'][-1], df_action)
                    df_value_2 = self.target_2(delayed_deque['obs'][-1], df_action)
                    df_value = torch.min(df_value_1, df_value_2)

                    if value < df_value:
                        action = action
                    else:
                        action = df_action

                    action = action.squeeze().cpu().numpy()
            
            next_obs, reward, done, info = self.env.step(action)

            delayed_deque['obs'].append(torch.FloatTensor(next_obs).to(self.config['device']))
            delayed_deque['action'].append(action)
            delayed_deque['reward'].append((reward - self.reward_mean) / self.reward_std)

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
                    next_latent = self.get_next_latent_trans_dy(next_latent, delayed_deque['action'], delayed_deque['reward'])
                else:
                    raise NotImplementedError
                next_rec_obs = self.auto_encoder.decode(next_latent)

            self.replay_buffer.store(obs, rec_obs, latent, action, reward, next_obs, next_rec_obs, next_latent, done)

            latent = next_latent
            obs = next_obs
            rec_obs = next_rec_obs
            if done:
                # self.log_metric.update(info['episode'])
                obs = self.env.reset()
                with torch.no_grad():
                    latent = self.auto_encoder.encode(torch.FloatTensor(obs).to(self.config['device']))
                delayed_deque = {
                    'obs': deque(maxlen=self.config['delay'] + 1),
                    'action': deque(maxlen=self.config['delay']),
                    'reward': deque(maxlen=self.config['delay']),
                }
                delayed_deque['obs'].append(torch.FloatTensor(obs).to(self.config['device']))
                rec_obs = torch.FloatTensor(obs).to(self.config['device'])
            
            if self.global_step > self.config['learn_start']:
                b_obs, b_rec_obs, b_latent, b_action, b_reward, b_next_obs, b_next_rec_obs, b_next_latent, b_done = self.replay_buffer.sample(batch_size=self.config['batch_size'], device=self.config['device'])
                # critic
                with torch.no_grad():
                    p_next_action, p_next_log_pi, _ = self.actor.get_action(b_next_rec_obs)
                    target_1_next_val = self.target_1(b_next_obs, p_next_action) - self.alpha * p_next_log_pi
                    p_next_action, p_next_log_pi, _ = self.df_actor.get_action(b_next_obs)
                    target_2_next_val = self.target_2(b_next_obs, p_next_action) - self.df_alpha * p_next_log_pi
                    target_next_val = torch.min(target_1_next_val, target_2_next_val)
                    next_q_val = b_reward + (1 - b_done) * self.config['gamma'] * (target_next_val)
                # critic_1_val = self.critic_1(b_latent, b_action)
                # critic_2_val = self.critic_2(b_latent, b_action)
                critic_1_val = self.critic_1(b_obs, b_action)
                critic_2_val = self.critic_2(b_obs, b_action)
                loss_critic_1 = F.mse_loss(critic_1_val, next_q_val)
                loss_critic_2 = F.mse_loss(critic_2_val, next_q_val)
                loss_critic = loss_critic_1 + loss_critic_2
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.critic_optimizer.step()
                # self.log_metric['loss_critic'] = loss_critic.item()


                if random.random() < 0.5:
                    # actor
                    # p_action, p_log_prob, _ = self.actor.get_action(b_latent)
                    p_action, p_log_prob, _ = self.actor.get_action(b_rec_obs)
                    # critic_1_q_val = self.critic_1(b_latent, p_action)
                    # critic_2_q_val = self.critic_2(b_latent, p_action)
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
                        # _, p_log_prob, _ = self.actor.get_action(b_latent)
                        _, p_log_prob, _ = self.actor.get_action(b_rec_obs)
                    loss_alpha = (-self.log_alpha.exp() * (p_log_prob + self.target_entropy)).mean()
                    self.alpha_optimizer.zero_grad()
                    loss_alpha.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()
                    # self.log_metric['loss_alpha'] = loss_alpha.item()
                else:
                    p_action, p_log_prob, _ = self.df_actor.get_action(b_obs)
                    critic_1_q_val = self.critic_1(b_obs, p_action)
                    critic_2_q_val = self.critic_2(b_obs, p_action)
                    min_critic_q_val = torch.min(critic_1_q_val, critic_2_q_val)
                    loss_df_actor = ((self.df_alpha * p_log_prob) - min_critic_q_val).mean()
                    self.df_actor_optimizer.zero_grad()
                    loss_df_actor.backward()
                    self.df_actor_optimizer.step()
                    # self.log_metric['loss_actor'] = loss_actor.item()

                    # alpha
                    with torch.no_grad():
                        _, p_log_prob, _ = self.df_actor.get_action(b_obs)
                    loss_df_alpha = (-self.log_df_alpha.exp() * (p_log_prob + self.target_entropy)).mean()
                    self.df_alpha_optimizer.zero_grad()
                    loss_df_alpha.backward()
                    self.df_alpha_optimizer.step()
                    self.df_alpha = self.log_df_alpha.exp().item()

                for param, target_param in zip(self.critic_1.parameters(), self.target_1.parameters()):
                    target_param.data.copy_(self.config['soft_update_factor'] * param.data + (1 - self.config['soft_update_factor']) * target_param.data)
                for param, target_param in zip(self.critic_2.parameters(), self.target_2.parameters()):
                    target_param.data.copy_(self.config['soft_update_factor'] * param.data + (1 - self.config['soft_update_factor']) * target_param.data)

            if self.global_step % self.config['evaluate_freq'] == 0:
                self.evaluate()

            self.logging()
        self.logger.close()
    
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

    def get_next_latent_trans_dy(self, next_latent, actions, rewards):
        delayed_idx = len(actions) - 1
        next_latent = next_latent.unsqueeze(0)
        # timesteps = torch.arange(0, len(actions), dtype=torch.int32).to(self.config['device'])
        timesteps = torch.arange(0, self.config['delay'], dtype=torch.int32).to(self.config['device'])
        masks = torch.zeros(len(actions)).unsqueeze(0).to(self.config['device'])
        pad_masks = torch.ones(self.config['delay'] - len(actions)).unsqueeze(0).to(self.config['device'])
        masks = torch.concat((masks, pad_masks), dim=-1)
        action_dim = actions[0].shape[0]
        pad_actions = torch.zeros((1, self.config['delay'] - len(actions), action_dim)).to(self.config['device'])
        pad_rewards = torch.zeros((1, self.config['delay'] - len(actions))).to(self.config['device'])
        next_latent = self.latent_dynamic(
            latents=next_latent, 
            actions=torch.concat((torch.FloatTensor(np.array(list(actions))).unsqueeze(0).to(self.config['device']), pad_actions), dim=1),
            rewards=torch.concat((torch.FloatTensor(np.array(list(rewards))).unsqueeze(0).to(self.config['device']), pad_rewards), dim=1).unsqueeze(-1),
            timesteps=timesteps,
            masks=masks
        )
        next_latent = next_latent[:, delayed_idx, :].squeeze(0)
        return next_latent

    def evaluate(self):
        self.log_metric['eval_r'] = []
        self.log_metric['eval_l'] = []

        obs = self.eval_env.reset()
        with torch.no_grad():
            latent = self.auto_encoder.encode(torch.FloatTensor(obs).to(self.config['device']))
        delayed_deque = {
            'obs': deque(maxlen=self.config['delay'] + 1),
            'action': deque(maxlen=self.config['delay']),
            'reward': deque(maxlen=self.config['delay']),
        }
        delayed_deque['obs'].append(torch.FloatTensor(obs).to(self.config['device']))
        rec_obs = torch.FloatTensor(obs).to(self.config['device'])

        while len(self.log_metric['eval_r']) < 10:
            with torch.no_grad():
                # action, _, _ = self.actor.get_action(latent)
                action, _, _ = self.actor.get_action(rec_obs)
                action = action.squeeze().cpu().numpy()
            
            next_obs, reward, done, info = self.eval_env.step(action)

            delayed_deque['obs'].append(torch.FloatTensor(next_obs).to(self.config['device']))
            delayed_deque['action'].append(action)
            delayed_deque['reward'].append((reward - self.reward_mean) / self.reward_std)

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
                    next_latent = self.get_next_latent_trans_dy(next_latent, delayed_deque['action'], delayed_deque['reward'])
                else:
                    raise NotImplementedError
                next_rec_obs = self.auto_encoder.decode(next_latent)

            latent = next_latent
            obs = next_obs
            rec_obs = next_rec_obs
            if done:
                self.log_metric['eval_r'].append(info['episode']['r'])
                self.log_metric['eval_l'].append(info['episode']['l'])
                obs = self.eval_env.reset()
                with torch.no_grad():
                    latent = self.auto_encoder.encode(torch.FloatTensor(obs).to(self.config['device']))
                delayed_deque = {
                    'obs': deque(maxlen=self.config['delay'] + 1),
                    'action': deque(maxlen=self.config['delay']),
                    'reward': deque(maxlen=self.config['delay']),
                }
                delayed_deque['obs'].append(torch.FloatTensor(obs).to(self.config['device']))
                rec_obs = torch.FloatTensor(obs).to(self.config['device'])

        self.log_metric['eval_r'] = np.mean(self.log_metric['eval_r'])
        self.log_metric['eval_l'] = np.mean(self.log_metric['eval_l'])



    def logging(self):
        for k in self.log_metric.keys():
            self.logger.add_scalar(k, self.log_metric[k], global_step=self.global_step)
        self.log_metric = {}
        

        


if __name__ == "__main__":
    configs = {
        "env_name": ["halfcheetah", "hopper", "walker2d"],
        "device":  ["cuda" if torch.cuda.is_available() else "cpu"],
        "seed": list(range(5)),  # Sets Gym, PyTorch and Numpy seeds
        "gamma": [0.99],
        "total_step": [int(1e6)],  # Max time steps to run environment
        "buffer_size": [int(1e6)],  # Replay buffer size
        "batch_size": [256],  # Batch size for all networks
        "lr_actor": [3e-4],
        "lr_critic": [1e-3],
        "lr_alpha": [1e-3],
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
        "dynamic_type": ['trans'],
    }
    configs = get_configs(configs)
    for config in configs:
        
        config['exp_tag'] = f"new_logs_online/adrl_w_br_reward/{config['dynamic_type']}/{config['env_name']}/{config['delay']}/SEED_{config['seed']}"
        if os.path.exists(config['exp_tag']):
            continue
        trainer = Trainer(config)
        trainer.train()
    