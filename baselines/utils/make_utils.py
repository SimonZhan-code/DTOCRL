import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Deque
from utils import *
import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from rich import print
from collections import deque
from tqdm import tqdm, trange
import os
from itertools import product

TensorBatch = List[torch.Tensor]

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

def get_list_configs(config_lists):
    combinations = list(product(*config_lists.values()))
    list_configs = []
    for combo in combinations:
        config_dict = {key: value for key, value in zip(config_lists.keys(), combo)}
        list_configs.append(config_dict)
    return list_configs

def find_specific_exp_tag(exp_dict, exp_tag):
    contents = os.listdir(exp_dict)
    for item in contents:
        if exp_tag in item:
            print(f'has exp already: {item}')
            return True
    return False
