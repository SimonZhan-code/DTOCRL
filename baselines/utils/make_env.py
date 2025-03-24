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
TensorBatch = List[torch.Tensor]

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


@torch.no_grad()
def eval_actor(
    env: gym.Env, 
    actor: nn.Module, 
    delay_step: int, 
    initial_actions_buffer: Deque[np.array], 
    augment: bool, 
    device: str, 
    n_episodes: int, 
    seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        states_buffer = deque(maxlen=delay_step+1)
        states_buffer.append(state)
        actions_buffer = initial_actions_buffer
        while not done:
            if delay_step == 0:
                action = actor.act(state, device)
            else:
                if augment:
                    augment_state = np.concatenate((states_buffer[0], np.array(actions_buffer).reshape(-1)), axis=0)
                    action = actor.act(augment_state, device)
                else:
                    action = actor.act(states_buffer[0], device)
            state, reward, done, _ = env.step(action)
            states_buffer.append(state)
            actions_buffer.append(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)