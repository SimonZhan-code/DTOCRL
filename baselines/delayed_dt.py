import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union, Deque
from utils import *
import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from rich import print
# import wandb
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm, trange  # noqa
from collections import deque
import time
import wandb
import uuid

from utils.make_utils import soft_update, set_seed, get_list_configs, find_specific_exp_tag, wandb_init
from utils.make_env import wrap_env
from utils.make_data import ReplayBuffer, modify_reward, normalize_states, compute_mean_std
from utils.make_eval import eval_actor

# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_d4rl_trajectories(
    env_name: str, gamma: float = 1.0, 
    augment: bool = True, delay_step: int = 5, 
    initial_actions_buffer: Deque[np.ndarray] = deque(maxlen=1), 
    normalize_state: bool = True,
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = gym.make(env_name).get_dataset()
    if normalize_state:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1
    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )

    traj, traj_len = [], []

    states_buffer = deque(maxlen=delay_step+1)
    actions_buffer = initial_actions_buffer

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        states_buffer.append(dataset["observations"][i])
        if delay_step == 0:
            data_["observations"].append(dataset["observations"][i])
        else:
            if augment:
                data_["observations"].append(np.concatenate((states_buffer[0], np.array(actions_buffer).reshape(-1)), axis=0))
            else:
                data_["observations"].append(states_buffer[0])
        actions_buffer.append(dataset["actions"][i])

        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:            
            states_buffer = deque(maxlen=delay_step+1)
            actions_buffer = initial_actions_buffer

            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


class SequenceDataset(IterableDataset):
    def __init__(self, 
                 env_name: str, 
                 seq_len: int = 10, 
                 reward_scale: float = 1.0,
                 augment: bool = True, 
                 delay_step: int = 5, 
                 initial_actions_buffer: Deque[np.ndarray] = deque(maxlen=1)):
        self.dataset, info = load_d4rl_trajectories(env_name, 
                                                    gamma=1.0,
                                                    augment=augment,
                                                    delay_step=delay_step,
                                                    initial_actions_buffer=initial_actions_buffer,)
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        # states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 1::3]) * self.max_action
        return out


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    target_return: float,
    device: str = "cpu",    
    augment: bool = True, 
    delay_step: int = 5, 
    initial_actions_buffer: Deque[np.ndarray] = deque(maxlen=1),
) -> Tuple[float, float]:
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )
    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    state = env.reset()
    states_buffer = deque(maxlen=delay_step+1)
    states_buffer.append(state)
    actions_buffer = initial_actions_buffer
    if delay_step == 0:
        states[:, 0] = torch.as_tensor(state, device=device)
    else:
        if augment:
            states[:, 0] = torch.as_tensor(np.concatenate((states_buffer[0], np.array(actions_buffer).reshape(1,-1)), axis=1), device=device)
        else:        
            states[:, 0] = torch.as_tensor(states_buffer[0], device=device)

    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        predicted_actions = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len :],
            actions[:, : step + 1][:, -model.seq_len :],
            returns[:, : step + 1][:, -model.seq_len :],
            time_steps[:, : step + 1][:, -model.seq_len :],
        )
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)

        states_buffer.append(next_state)
        actions_buffer.append(predicted_action)

        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(np.concatenate((states_buffer[0], np.array(actions_buffer).reshape(1,-1)), axis=1), device=device)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if done:
            break

    return episode_return, episode_len

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

    set_seed(config["seed"])

    # evaluation environment with state & reward preprocessing (as in dataset above)
    env = gym.make(config["env"])

    initial_actions_buffer = deque(maxlen=config["delay_step"])
    for _ in range(config["delay_step"]):
        initial_actions_buffer.append(env.action_space.sample())

    # data & dataloader setup
    dataset = SequenceDataset(
        config["env"], 
        seq_len=config["seq_len"], 
        reward_scale=config["reward_scale"],
        augment=config["augment"],
        delay_step=config["delay_step"],
        initial_actions_buffer=initial_actions_buffer,
    )
    trainloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        pin_memory=True,
        num_workers=config["num_workers"],
    )

    eval_env = wrap_env(
        env=env,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config["reward_scale"],
    )
    # model & optimizer & scheduler setup
    config["state_dim"] = eval_env.observation_space.shape[0]
    config["action_dim"] = eval_env.action_space.shape[0]
    if config["augment"]:
        model = DecisionTransformer(
            state_dim=config["state_dim"] + config["delay_step"] * config["action_dim"],
            action_dim=config["action_dim"],
            embedding_dim=config["embedding_dim"],
            seq_len=config["seq_len"],
            episode_len=config["episode_len"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            attention_dropout=config["attention_dropout"],
            residual_dropout=config["residual_dropout"],
            embedding_dropout=config["embedding_dropout"],
            max_action=config["max_action"],
        ).to(config["device"])
    else:
        model = DecisionTransformer(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            embedding_dim=config["embedding_dim"],
            seq_len=config["seq_len"],
            episode_len=config["episode_len"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            attention_dropout=config["attention_dropout"],
            residual_dropout=config["residual_dropout"],
            embedding_dropout=config["embedding_dropout"],
            max_action=config["max_action"],
        ).to(config["device"])

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config["warmup_steps"], 1),
    )
    # save config to the checkpoint
    if config["checkpoints_path"] is not None:
        print(f'Checkpoints path: {config["checkpoints_path"]}')
        os.makedirs(config["checkpoints_path"], exist_ok=True)
        with open(os.path.join(config["checkpoints_path"], "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)

    wandb_init(config)

    for step in trange(config["update_steps"], desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config["device"]) for b in batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (loss * mask.unsqueeze(-1)).mean()

        optim.zero_grad()
        loss.backward()
        if config["clip_grad"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
        optim.step()
        scheduler.step()

        # validation in the env for the actual online performance
        if step % config["eval_every"] == 0 or step == config["update_steps"] - 1:
            model.eval()
            normalized_scores = []
            episode_len = []
            for target_return in config["target_returns"]:
                eval_env.seed(config["seed"])
                eval_returns, eval_lens = [], []
                for _ in trange(config["eval_episodes"], desc="Evaluation", leave=False):
                    eval_return, eval_len = eval_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * config["reward_scale"],
                        device=config["device"],
                        augment=config["augment"],
                        delay_step=config["delay_step"],
                        initial_actions_buffer=initial_actions_buffer,
                    )
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config["reward_scale"])
                    eval_lens.append(eval_len)
                normalized_scores.append(np.mean(eval_env.get_normalized_score(np.array(eval_returns))))
                episode_len.append(np.mean(eval_lens))
            logger.add_scalar("eval/normalized_score", np.max(normalized_scores), step)
            logger.add_scalar("eval/episode_len", np.maximum(episode_len), step)
            print(np.max(normalized_scores))
            wandb.log({
                "eval_r": np.max(normalized_scores), 
                "eval_l": np.max(episode_len),
                })
            model.train()

    if config["checkpoints_path"] is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(checkpoint, os.path.join(config["checkpoints_path"], "dt_checkpoint.pt"))
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
        # Decision Transformer
        "embedding_dim": [128],
        "num_layers": [3],
        "num_heads": [1],
        "seq_len": [20],
        "episode_len": [1000],
        "attention_dropout": [0.1],
        "residual_dropout": [0.1],
        "embedding_dropout": [0.1],
        "max_action": [1.0],
        "learning_rate": [1e-4],
        "betas": [(0.9, 0.999)],
        "weight_decay": [1e-4],
        "clip_grad": [0.25],
        "batch_size": [64],
        "update_steps": [int(1e6)],
        "warmup_steps": [10_000],
        "reward_scale": [0.001],
        "num_workers": [4],
        "target_returns": [(12000.0, 6000.0)],
        "eval_episodes": [10],
        "eval_every": [1e4],
        "checkpoints_path": [None],
        # Delay
        "delay_step":[8],
        "augment": [True],
        # Wandb Naming
        "project": ["Offline_Delayed_RL"],
        "group": ["Delayed-DT"],
        "name": ["DT-D4RL"]
    }
    list_configs = get_list_configs(config_lists)
    for config in tqdm(list_configs):
        if config["augment"]:
            config["name"] = f'Augmented-{config["exp_name"]}-{config["env"]}-delay_step={config["delay_step"]}-seed={config["seed"]}-{str(uuid.uuid4())[:8]}'
        else:
            config["name"] = f'Memoryless-{config["exp_name"]}-{config["env"]}-delay_step={config["delay_step"]}-seed={config["seed"]}-{str(uuid.uuid4())[:8]}'
        if config["checkpoints_path"] is not None:
            config["checkpoints_path"] = os.path.join(config["checkpoints_path"], config["name"])
        print(config)

        train(config)
