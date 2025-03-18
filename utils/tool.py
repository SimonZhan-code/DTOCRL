import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import itertools
from itertools import product
import torch

def get_offline_dataset(env=None, policy=None):
    if env is None:
        envs = ['ant', 'halfcheetah', 'hopper', 'walker2d']
    else:
        envs = [env]
    if policy is None:
        policies = ['random', 'medium', 'expert', 'medium-replay', 'medium-expert']
    else:
        policies = [policy]
    dataset = []
    for env in envs:
        for policy in policies:
            dataset.append(f'{env}-{policy}-v2')
    return dataset

def get_auto_encoder(dataset_name, latent_dim):
    from utils.dataset_env import make_replay_buffer_env
    from utils.network import AutoEncoder

    _, env = make_replay_buffer_env(dataset_name)
    state_dim = env.observation_space.shape[0]
    auto_encoder = AutoEncoder(input_dim=state_dim, hidden_dim=256, latent_dim=latent_dim).to("cpu")
    checkpoint = torch.load(f"checkpoints/auto_encoder/D_{dataset_name}_L_{latent_dim}.pth", map_location=torch.device('cpu'))
    auto_encoder.load_state_dict(checkpoint['auto_encoder'])
    auto_encoder.eval()
    return auto_encoder

def get_bisimulation_encoder(dataset_name, latent_dim):
    from utils.dataset_env import make_replay_buffer_env
    from utils.network import AutoEncoder

    _, env = make_replay_buffer_env(dataset_name)
    state_dim = env.observation_space.shape[0]
    auto_encoder = AutoEncoder(input_dim=state_dim, hidden_dim=256, latent_dim=latent_dim).to("cpu")
    checkpoint = torch.load(f"checkpoints/bisimulation/D_{dataset_name}_L_{latent_dim}.pth", map_location=torch.device('cpu'))
    auto_encoder.load_state_dict(checkpoint)
    auto_encoder.eval()
    return auto_encoder

def get_latent_dynamic(dataset_name, dynamic_type, latent_dim, delay):
    from utils.dataset_env import make_replay_buffer_env
    from utils.network import MLP_Dynamic, GRU_Dynamic, TRANS_Dynamic

    _, env = make_replay_buffer_env(dataset_name)
    action_dim = env.action_space.shape[0]
    checkpoint = torch.load(f"checkpoints/{dynamic_type}_dynamic/D_{dataset_name}_L_{latent_dim}_Delays_{delay}.pth", map_location=torch.device('cpu'))
    if dynamic_type == 'mlp':
        latent_dynamic = MLP_Dynamic(latent_dim=latent_dim, condition_dim=action_dim, hidden_dim=latent_dim).to("cpu")
        # checkpoint = torch.load(f"checkpoints/{dynamic_type}_dynamic/{dataset_name}_L_{latent_dim}_Delays_{delay}.pth", map_location=torch.device('cpu'))
    elif dynamic_type == 'gru':
        latent_dynamic = GRU_Dynamic(latent_dim=latent_dim, condition_dim=action_dim, hidden_dim=latent_dim).to("cpu")
        # checkpoint = torch.load(f"checkpoints/{dynamic_type}_dynamic/{dataset_name}_L_{latent_dim}_Delays_{delay}.pth", map_location=torch.device('cpu'))
    elif dynamic_type == 'trans':
        latent_dynamic = TRANS_Dynamic(latent_dim=latent_dim, 
                                       condition_dim=action_dim, 
                                       seq_len=delay, 
                                       hidden_dim=latent_dim).to("cpu")
    else:
        raise NotImplementedError
    # checkpoint = torch.load(f"checkpoints/{dynamic_type}_dynamic/{dataset_name}_D_{delay}_L_{latent_dim}.pth", map_location=torch.device('cpu'))
    latent_dynamic.load_state_dict(checkpoint['latent_dynamic'])
    latent_dynamic.eval()
    return latent_dynamic


def get_configs(configs):
    combinations = list(product(*configs.values()))
    list_configs = []
    for combo in combinations:
        config_dict = {key: value for key, value in zip(configs.keys(), combo)}
        list_configs.append(config_dict)
    return list_configs