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
# import wandb
from tensorboardX import SummaryWriter
from rich import print
from collections import deque
from tqdm import tqdm, trange
import time, wandb


from utils.make_utils import soft_update, compute_mean_std, normalize_states, set_seed, get_list_configs, find_specific_exp_tag, wandb_init
from utils.make_env import wrap_env
from utils.make_data import ReplayBuffer, keep_best_trajectories
from utils.make_eval import eval_actor
from utils.make_nn import Actor

TensorBatch = List[torch.Tensor]
class BC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


def train(config):
    
    if not os.path.exists(f'runs/{config["exp_name"]}'):
        os.makedirs(f'runs/{config["exp_name"]}')
    
    exp_tag = f'{config["env"]}_DELAY_STEPS_{config["delay_step"]}_SEED_{config["seed"]}'
    if find_specific_exp_tag(f'runs/{config["exp_name"]}', exp_tag):
        return
    exp_tag += f'_{int(time.time())}'

    logger = SummaryWriter(f'runs/{config["exp_name"]}/{exp_tag}')
    logger.add_text(
        "config",
        "|parametrix|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    env = gym.make(config["env"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    # keep_best_trajectories(dataset, config["frac"], config["discount"])

    if config["normalize"]:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    if config["augment"]:
        replay_buffer = ReplayBuffer(
            state_dim + config["delay_step"] * action_dim,
            action_dim,
            config["buffer_size"],
            config["device"],
        )
    else:
        replay_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            config["buffer_size"],
            config["device"],
        )

    initial_actions_buffer = deque(maxlen=config["delay_step"])
    for _ in range(config["delay_step"]):
        initial_actions_buffer.append(env.action_space.sample())
    replay_buffer.load_d4rl_dataset(dataset, config["delay_step"], initial_actions_buffer, config["augment"])

    if config["checkpoints_path"] is not None:
        print(f'Checkpoints path: {config["checkpoints_path"]}')
        os.makedirs(config["checkpoints_path"], exist_ok=True)
        with open(os.path.join(config["checkpoints_path"], "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = config["seed"]
    set_seed(seed, env)

    if config["augment"]:
        actor = Actor(state_dim + config["delay_step"] * action_dim, action_dim, max_action).to(config["device"])
    else:
        actor = Actor(state_dim, action_dim, max_action).to(config["device"])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["lr"])

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config["discount"],
        "device": config["device"],
    }

    print("---------------------------------------")
    print(f'Training BC, Env: {config["env"]}, Seed: {seed}')
    print("---------------------------------------")

    wandb_init(config)

    # Initialize policy
    trainer = BC(**kwargs)

    if config["load_model"] != "":
        policy_file = Path(config["load_model"])
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    evaluations = []
    for t in trange(int(config["max_timesteps"])):
        batch = replay_buffer.sample(config["batch_size"])
        batch = [b.to(config["device"]) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict)
        # Evaluate episode
        if (t + 1) % config["eval_freq"] == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, eval_length = eval_actor(
                env,
                actor,
                delay_step=config["delay_step"],
                initial_actions_buffer=initial_actions_buffer,
                augment=config["augment"],
                device=config["device"],
                n_episodes=config["n_episodes"],
                seed=config["seed"],
            )
            eval_score = eval_scores.mean()
            eval_std = eval_scores.std()
            eval_len = eval_length.mean()
            normalized_eval_std = env.get_normalized_score(eval_std)
            normalized_eval_score = env.get_normalized_score(eval_score)
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f'Evaluation over {config["n_episodes"]} episodes: '
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")

            if config["checkpoints_path"] is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config["checkpoints_path"], f"checkpoint_{t}.pt"),
                )

            logger.add_scalar("eval/eval_r", normalized_eval_score, int(t))
            logger.add_scalar("eval/eval_r_std", normalized_eval_std, int(t))
            logger.add_scalar("training/eval_len", eval_len, int(t))
            wandb.log({
                "eval_r": normalized_eval_score, 
                "eval_r_std": normalized_eval_std,
                "eval_l": eval_len,
            })

    logger.close()
    wandb.finish()

if __name__ == "__main__":
    config_lists = {
        "exp_name": [os.path.basename(__file__).rstrip(".py")],
        # Experiment
        "device":  ["cuda" if torch.cuda.is_available() else "cpu"],
        "env": [
                "hopper-medium-v2", 
                "hopper-expert-v2",
                "hopper-medium-expert-v2",
                "hopper-medium-replay-v2",

                "halfcheetah-medium-v2", 
                "halfcheetah-expert-v2",
                "halfcheetah-medium-expert-v2",
                "halfcheetah-medium-replay-v2",
            
                "walker2d-medium-v2", 
                "walker2d-expert-v2",
                "walker2d-medium-expert-v2",
                "walker2d-medium-replay-v2",
                ],  # OpenAI gym environment name
        "seed": [1, 2, 3],  # Sets Gym, PyTorch and Numpy seeds
        "eval_freq": [int(1e4)],  # How often (time steps) we evaluate
        "n_episodes": [10],  # How many episodes run during evaluation
        "max_timesteps": [int(1e6)],  # Max time steps to run environment
        "checkpoints_path": [None],  # Save path
        "load_model": [""],  # Model load file name, "" doesn't load
        "batch_size": [256],  # Batch size for all networks
        "discount": [0.99],  # Discount factor
        # BC
        "buffer_size": [2_000_000],  # Replay buffer size
        "frac": [0.1],  # Best data fraction to use
        "max_traj_len": [1000],  # Max trajectory length
        "normalize": [True],  # Normalize states
        "lr": [3e-4], # learning rate
        # Delay
        "delay_step":[8],
        "augment": [True],
        # Wandb logging
        "project": ["Offline_Delayed_RL"],
        "group": ["Delayed-BC"],
        "name": ["BC-D4RL"]
    }
    list_configs = get_list_configs(config_lists)
    for config in tqdm(list_configs):
        if config["augment"]:
            config["name"] = f'Augmented-{config["exp_name"]}-{config["env"]}-delay_step={config["delay_step"]}-seed={config["seed"]}-{str(uuid.uuid4())[:8]}'
        else:
            config["name"] = f'Memoryless-{config["exp_name"]}-{config["env"]}-delay_step={config["delay_step"]}-seed={config["seed"]}-{str(uuid.uuid4())[:8]}'
        if config["checkpoints_path"] is not None:
            config["checkpoints_path"] = os.path.join(config["checkpoints_path"], config["exp_name"])
        print(config)

        train(config)
