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
    episode_rewards, episode_length = [], []
    for _ in range(n_episodes):
        state, done, length = env.reset(), False, 0
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
            length += 1
        episode_rewards.append(episode_reward)
        episode_length.append(length)

    actor.train()
    return np.asarray(episode_rewards), np.asarray(episode_length)



@torch.no_grad()
def eval_actor_random(
    env: gym.Env, 
    actor: nn.Module, 
    max_obs_delay_step: int,
    delay_prior: torch.distributions.Distribution, 
    initial_actions_buffer: Deque[np.array],
    augment: bool, 
    device: str, 
    n_episodes: int, 
    seed: int
) -> np.ndarray:
    env.reset(seed=seed)
    actor.eval()
    episode_rewards = []
    aug_dim = env.observation_space.shape[0] + env.action_space.shape[0] * max_obs_delay_step
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        states_buffer = deque(maxlen=max_obs_delay_step+1)
        states_buffer.append(state)
        actions_buffer = initial_actions_buffer
        while not done:
            delay_step = np.clip(delay_prior.sample().int().item(), 0, max_obs_delay_step)
            if max_obs_delay_step == 0:
                action = actor.act(state, device)
            else:
                if augment:
                    try:
                        augment_state = np.zeros(aug_dim)
                        temp_state = np.concatenate((states_buffer[-delay_step-1],\
                                                         np.array(actions_buffer[-delay_step-1:]).reshape(-1)), axis=0)
                        augment_state[:len(temp_state)] = temp_state
                        action = actor.act(augment_state, device)
                    except:
                        augment_state = np.concatenate((states_buffer[0], np.array(actions_buffer).reshape(-1)), axis=0)
                        action = actor.act(augment_state, device)
                else:
                    try:
                        action = actor.act(states_buffer[-delay_step-1], device)
                    except:
                        action = actor.act(states_buffer[0], device)
            state, reward, done, _ = env.step(action)
            states_buffer.append(state)
            actions_buffer.append(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)

