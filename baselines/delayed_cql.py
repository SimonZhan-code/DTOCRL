import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Deque
from utils import *
import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
# import wandb
from tensorboardX import SummaryWriter
from rich import print
from collections import deque
from tqdm import tqdm, trange
import time
import wandb

from utils.make_utils import soft_update, compute_mean_std, normalize_states, set_seed, get_list_configs, find_specific_exp_tag, wandb_init
from utils.make_env import wrap_env
from utils.make_data import ReplayBuffer, modify_reward
from utils.make_eval import eval_actor
from utils.make_nn import Scalar, FullyConnectedQFunction, TanhGaussianPolicy

TensorBatch = List[torch.Tensor]

class ContinuousCQL:
    def __init__(
        self,
        critic_1,
        critic_1_optimizer,
        critic_2,
        critic_2_optimizer,
        actor,
        actor_optimizer,
        target_entropy: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cpu",
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        self.total_it += 1

        new_actions, log_pi = self.actor(observations)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(
            observations, actions, new_actions, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            observations, actions, next_observations, rewards, dones, alpha, log_dict
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
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

    if config["normalize_reward"]:
        modify_reward(
            dataset,
            config["env"],
            reward_scale=config["reward_scale"],
            reward_bias=config["reward_bias"],
        )

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
        critic_1 = FullyConnectedQFunction(
            state_dim + config["delay_step"] * action_dim,
            action_dim,
            config["orthogonal_init"],
            config["q_n_hidden_layers"],
        ).to(config["device"])
        critic_2 = FullyConnectedQFunction(
            state_dim + config["delay_step"] * action_dim, 
            action_dim, 
            config["orthogonal_init"]
        ).to(config["device"])
    else:
        critic_1 = FullyConnectedQFunction(
            state_dim,
            action_dim,
            config["orthogonal_init"],
            config["q_n_hidden_layers"],
        ).to(config["device"])
        critic_2 = FullyConnectedQFunction(
            state_dim, 
            action_dim, 
            config["orthogonal_init"]
        ).to(config["device"])
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config["qf_lr"])
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config["qf_lr"])

    if config["augment"]:
        actor = TanhGaussianPolicy(
            state_dim + config["delay_step"] * action_dim,
            action_dim,
            max_action,
            log_std_multiplier=config["policy_log_std_multiplier"],
            orthogonal_init=config["orthogonal_init"],
        ).to(config["device"])
    else:
        actor = TanhGaussianPolicy(
            state_dim,
            action_dim,
            max_action,
            log_std_multiplier=config["policy_log_std_multiplier"],
            orthogonal_init=config["orthogonal_init"],
        ).to(config["device"])
    actor_optimizer = torch.optim.Adam(actor.parameters(), config["policy_lr"])

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config["discount"],
        "soft_target_update_rate": config["soft_target_update_rate"],
        "device": config["device"],
        # CQL
        "target_entropy": -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": config["alpha_multiplier"],
        "use_automatic_entropy_tuning": config["use_automatic_entropy_tuning"],
        "backup_entropy": config["backup_entropy"],
        "policy_lr": config["policy_lr"],
        "qf_lr": config["qf_lr"],
        "bc_steps": config["bc_steps"],
        "target_update_period": config["target_update_period"],
        "cql_n_actions": config["cql_n_actions"],
        "cql_importance_sample": config["cql_importance_sample"],
        "cql_lagrange": config["cql_lagrange"],
        "cql_target_action_gap": config["cql_target_action_gap"],
        "cql_temp": config["cql_temp"],
        "cql_alpha": config["cql_alpha"],
        "cql_max_target_backup": config["cql_max_target_backup"],
        "cql_clip_diff_min": config["cql_clip_diff_min"],
        "cql_clip_diff_max": config["cql_clip_diff_max"],
    }

    
    print("---------------------------------------")
    print(f'Training CQL, Env: {config["env"]}, Seed: {seed}')
    print("---------------------------------------")

    # Initialize actor
    trainer = ContinuousCQL(**kwargs)


    if config["load_model"] != "":
        policy_file = Path(config["load_model"])
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(config)
    
    evaluations = []
    for t in trange(int(config["max_timesteps"])):
        batch = replay_buffer.sample(config["batch_size"])
        batch = [b.to(config["device"]) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=int(t))
        # Evaluate episode
        if (t + 1) % config["eval_freq"] == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
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
            normalized_eval_std = env.get_normalized_score(eval_std) * 100.0
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
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

            logger.add_scalar("training/normalized_score", 
                              normalized_eval_score, 
                              int(trainer.total_it))
            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score,
                 "d4rl_normalized_std": normalized_eval_std}, step=trainer.total_it
            )

    logger.close()


if __name__ == "__main__":
    config_lists = {
        "exp_name": [os.path.basename(__file__).rstrip(".py")],
        # Experiment
        "device":  ["cuda" if torch.cuda.is_available() else "cpu"],
        "env": [
                # "halfcheetah-medium-v2", 
                # "halfcheetah-medium-expert-v2",
                # "halfcheetah-expert-v2",

                "hopper-medium-v2",
                # "hopper-medium-expert-v2",
                # "hopper-expert-v2",

                # "walker2d-medium-v2",
                # "walker2d-medium-expert-v2",
                # "walker2d-expert-v2",
                ],  # OpenAI gym environment name
        "seed": [1],  # Sets Gym, PyTorch and Numpy seeds
        "eval_freq": [int(1e4)],  # How often (time steps) we evaluate
        "n_episodes": [10],  # How many episodes run during evaluation
        "max_timesteps": [int(1e6)],  # Max time steps to run environment
        "checkpoints_path": [None],  # Save path
        "load_model": [""],  # Model load file name, "" doesn't load
        "buffer_size": [2_000_000],  # Replay buffer size
        "batch_size": [256],  # Batch size for all networks
        "discount": [0.99],  # Discount factor
        # CQL
        "alpha_multiplier": [1.0],# Multiplier for alpha in loss
        "use_automatic_entropy_tuning": [True],  # Tune entropy
        "backup_entropy": [False],  # Use backup entropy

        "policy_lr": [3e-5], # Policy learning rate
        "qf_lr": [3e-4], # Critics learning rate
        "soft_target_update_rate": [5e-3], # Target network update rate
        "target_update_period": [int(1)], # Frequency of target nets updates
        "cql_n_actions": [int(10)], # Number of sampled actions
        "cql_importance_sample": [True],  # Use importance sampling
        "cql_lagrange": [False],  # Use Lagrange version of CQL
        "cql_target_action_gap": [-1.0],  # Action gap
        "cql_temp": [1.0],  # CQL temperature
        "cql_alpha": [10.0],  # Minimal Q weight
        "cql_max_target_backup": [False],  # Use max target backup
        "cql_clip_diff_min": [-np.inf],  # Q-function lower loss clipping
        "cql_clip_diff_max": [np.inf],  # Q-function upper loss clipping
        "orthogonal_init": [True],  # Orthogonal initialization
        "normalize": [True],  # Normalize states
        "normalize_reward": [False],  # Normalize reward
        "q_n_hidden_layers": [int(3)],  # Number of hidden layers in Q networks
        "reward_scale": [1.0],  # Reward scale for normalization
        "reward_bias": [0.0],  # Reward bias for normalization
        "bc_steps": [int(0)],
        "policy_log_std_multiplier": [1.0],
        # Delay
        "delay_step":[8],
        "augment": [True],
        # Wandb logging
        "project": ["Delayed-Offline-RL"],
        "group": ["Aug-CQL-D4RL"],
        "name": ["Delayed-CQL"]
    }
    list_configs = get_list_configs(config_lists)
    for config in tqdm(list_configs):
        if config["augment"]:
            config["name"] = f'Augmented-{config["name"]}-{config["env"]}-delay_step={config["delay_step"]}-seed={config["seed"]}-{str(uuid.uuid4())[:8]}'
        else:
            config["name"] = f'Memoryless-{config["name"]}-{config["env"]}-delay_step={config["delay_step"]}-seed={config["seed"]}-{str(uuid.uuid4())[:8]}'
        if config["checkpoints_path"] is not None:
            config["checkpoints_path"] = os.path.join(config["checkpoints_path"], config["name"])
        print(config)

        train(config)