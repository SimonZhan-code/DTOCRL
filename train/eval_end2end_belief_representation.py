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
import pickle

class Evaluator():
    def __init__(self, config):
        self.config = config
        # self.logger = SummaryWriter(config['exp_tag'])
        # self.logger.add_text(
        #     "config",
        #     "|parametrix|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
        # )
        self.log_dict = {}


        env = gym.make(f"{config['env_name']}-random-v2")
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # load replay_buffer
        self.replay_buffer = DelayBuffer(
            self.observation_dim, 
            action_dim=self.action_dim,
            delay=self.config['delay'],
        )
        # for policy in ['random', 'medium', 'expert']:
        for policy in ['expert']:
            dataset_name = f"{config['env_name']}-{policy}-v2"
            self.replay_buffer.load_d4rl_dataset(dataset_name)
        # self.replay_buffer.normalize_reward()

        self.auto_encoder = AutoEncoder(
            input_dim=self.observation_dim, 
            hidden_dim=256, 
            latent_dim=self.config['latent_dim']).to("cuda")
        # checkpoint = torch.load(f"new_checkpoints/auto_encoder/{self.config['env_name']}.pth", map_location=torch.device('cuda'))
        # checkpoint = torch.load(f"new_checkpoints/end2end/{self.config['env_name']}_{self.config['dynamic_type']}_Delay_{self.config['delay']}.pth", map_location=torch.device('cuda'))
        checkpoint = torch.load(f"new_checkpoints/trans_reward/{self.config['env_name']}_{self.config['dynamic_type']}_Delay_{self.config['delay']}.pth", map_location=torch.device('cuda'))

        self.auto_encoder.load_state_dict(checkpoint['auto_encoder'])
        self.auto_encoder.eval()
        print('loaded auto encoder')

        self.reward_mean = checkpoint['reward_mean']
        self.reward_std = checkpoint['reward_std']

    def eval(self):
        for self.config['dynamic_type'] in ['trans']:
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
            # checkpoint = torch.load(f"new_checkpoints/end2end/{self.config['env_name']}_{self.config['dynamic_type']}_Delay_{self.config['delay']}.pth", map_location=torch.device('cuda'))
            checkpoint = torch.load(f"new_checkpoints/trans_reward/{self.config['env_name']}_{self.config['dynamic_type']}_Delay_{self.config['delay']}.pth", map_location=torch.device('cuda'))
            print(self.config['dynamic_type'])
            print(self.config['env_name'])
            print(checkpoint['step'])
            # return 0, 0
            # exit()
            self.dynamic.load_state_dict(checkpoint['latent_dynamic'])
            self.dynamic.eval()
            print('loaded latent dynamic')


            # self.replay_buffer.generate_sample_prior()
            if self.config['dynamic_type'] == 'trans':
                self.inference_trans_dynamic()
                # pred_error_mean, pred_error_std = self.eval_trans_dynamic()
                
            else:
                raise NotImplementedError


        #     result = {
        #         'mean': pred_error_mean,
        #         'std': pred_error_std,
        #     }
        #     with open(f'new_checkpoints/{self.config["env_name"]}_Delay_{self.config["delay"]}_{self.config["dynamic_type"]}_reward.pkl', 'wb') as file:
        #         pickle.dump(result, file)
        #         file.close()

                
        # return pred_error_mean, pred_error_std
        


    def eval_trans_dynamic(self):
        pred_errors = [None for _ in range(self.config['delay'])]
        for indices in tqdm(self.replay_buffer._sample_prior):
            states, actions, rewards, dones, masks = self.replay_buffer.sample(indices)
            states = states.to(self.config['device'])
            actions = actions.to(self.config['device'])
            rewards = rewards.to(self.config['device'])
            masks = masks[:, 1:, 0].to(self.config['device'])

            with torch.no_grad():
                _, latents = self.auto_encoder(states)
                timesteps = torch.arange(0, self.config['delay'], dtype=torch.int32).to(self.config['device'])
                z = self.dynamic(latents=latents[:, :1, :], 
                                actions=actions[:, :self.config['delay'], :],
                                rewards=rewards[:, :self.config['delay'], :],
                                timesteps=timesteps,
                                masks=masks)
                states_pred = self.auto_encoder.decode(z)
            errors = F.l1_loss(states[:, 1:, :], states_pred, reduction='none').mean(-1)
            masks = 1 - masks
            for i in range(self.config['delay']):
                idx = torch.where(masks[:, i] == 1)
                if pred_errors[i] == None:
                    pred_errors[i] = errors[:, i][idx]
                else:
                    pred_errors[i] = torch.cat((pred_errors[i], errors[:, i][idx]), dim=0)
        means = []
        stds = []
        for i in range(self.config['delay']):
            means.append(pred_errors[i].mean().item())
            stds.append(pred_errors[i].std().item())
        return means, stds

    def inference_trans_dynamic(self):
        
        indices = np.array(range(100))
        # indices = self.replay_buffer._sample_prior[0]
        states, actions, rewards, dones, masks = self.replay_buffer.sample(indices)
        states = states.to(self.config['device'])
        actions = actions.to(self.config['device'])
        rewards = (rewards.to(self.config['device']) - self.reward_mean) / self.reward_std
        masks = masks[:, 1:, 0].to(self.config['device'])

        with torch.no_grad():
            _, latents = self.auto_encoder(states)
            timesteps = torch.arange(0, self.config['delay'], dtype=torch.int32).to(self.config['device'])
            z = self.dynamic(latents=latents[:, :1, :], 
                            actions=actions[:, :self.config['delay'], :],
                            rewards=rewards[:, :self.config['delay'], :],
                            timesteps=timesteps,
                            masks=masks)
            states_pred = self.auto_encoder.decode(z)
        pred_states = torch.concat((states[:, :1, :], states_pred), dim=1)
        print(pred_states.shape)
        with open(f'belief_inference_rollout/{self.config["env_name"]}_dbr_inference_rollout.pkl', 'wb') as file:
            pickle.dump({
                'ground_truth': states.cpu().numpy(),
                'belief_estimation':pred_states.cpu().numpy(),
            }, file)
            file.close()

        return


    def logging(self):
        for k in self.log_dict.keys():
            self.logger.add_scalar(k, self.log_dict[k], global_step=self.global_step)
        self.log_dict = {}


import matplotlib.pyplot as plt
import seaborn as sns
def plot_pickle(env):
    plt.figure(figsize=(8, 6))
    for dynamic_type in ['Gaussian (DATS)', 'GRU (DEER)', 'Transformer (D-SAC)', 'DBT (ours)']:
        file = {
            'Gaussian (DATS)': 'mlp',
            'GRU (DEER)': 'gru',
            'Transformer (D-SAC)': 'trans',
            'DBT (ours)': 'trans_reward',
        }
        with open(f'new_checkpoints/{env}_Delay_128_{file[dynamic_type]}.pkl', 'rb') as file:
            data = pickle.load(file)
        mean = np.array(data['mean'])
        std = np.array(data['std'])
        x = np.arange(1, len(mean) + 1)
        sns.lineplot(x=x, y=mean, label=f'{dynamic_type}', linewidth=2.5)
        plt.fill_between(x, mean - std, mean + std, alpha=0.1)
    plt.legend(fontsize=18)
    plt.xlim(0, 130)
    plt.ylim(0, 2)
    plt.subplots_adjust(left=0.11,    # left side of the plot
                        right=0.99,    # right side
                        bottom=0.11,   # bottom margin
                        top=0.99)      # top margin
    plt.xticks([2, 4, 8, 16, 32, 64, 128])
    plt.xlabel('Delays', fontsize=25)
    plt.ylabel('Belief Error (L1 Norm)', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{env}.pdf')

def check_pth():
    dir = 'new_checkpoints/latent_dynamic/'
    files = os.listdir(dir)
    files = [f for f in files if f.endswith('.pth')]
    for file in files:
        checkpoint = torch.load(dir + file, map_location=torch.device('cpu'))
        print(file)
        print(checkpoint['step'])



if __name__ == "__main__":
    # for env in ["halfcheetah", "hopper", "walker2d"]:
    #     plot_pickle(env)
    # check_pth()
    # exit()
    configs = {
        "env_name": ["halfcheetah", "hopper", "walker2d"],
        "device":  ["cuda"],
        "delay": [128],
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
        evaluator = Evaluator(config)
        evaluator.eval()