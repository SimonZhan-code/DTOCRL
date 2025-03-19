import d4rl
import gym
import numpy as np
import torch
from typing import List, Dict, Deque, Callable, Union
from tqdm import trange
from rich import print
import gym
from collections import deque
from torch.utils.data import Dataset
TensorBatch = List[torch.Tensor]

def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif value is None:
            result[key] = None
        else:
            result[key] = func(value)
    return result


def compute_mean_std(data, eps=1e-3):
    mean = data.mean(0)
    std = data.std(0) + eps
    return mean, std


def normalize_data(data, mean, std):
    return (data - mean) / std


def wrap_env(env, state_mean, state_std, reward_scale=1.0):
    def normalize_state(state):
        return (state - state_mean) / state_std
    def scale_reward(reward):
        return reward_scale * reward
    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.state_mean = state_mean
    env.state_std = state_std
    
    return env


def at_least_ndim(x: Union[np.ndarray, torch.Tensor, int, float], ndim: int, pad: int = 0):
    """ Add dimensions to the input tensor to make it at least ndim-dimensional.

    Args:
        x: Union[np.ndarray, torch.Tensor, int, float], input tensor
        ndim: int, minimum number of dimensions
        pad: int, padding direction. `0`: pad in the last dimension, `1`: pad in the first dimension

    Returns:
        Any of these 2 options

        - np.ndarray or torch.Tensor: reshaped tensor
        - int or float: input value

    Examples:
        >>> x = np.random.rand(3, 4)
        >>> at_least_ndim(x, 3, 0).shape
        (3, 4, 1)
        >>> x = torch.randn(3, 4)
        >>> at_least_ndim(x, 4, 1).shape
        (1, 1, 3, 4)
        >>> x = 1
        >>> at_least_ndim(x, 3)
        1
    """
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            if pad == 0:
                return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            if pad == 0:
                return torch.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return torch.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device


    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)


    def sample(self, indices=None, batch_size=256) -> TensorBatch:
        if indices is None:
            indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return states, actions, rewards, next_states, dones
    

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError
    
    def normalize_reward(self):
        self.reward_mean = self._rewards[:self._size].mean()
        self.reward_std = self._rewards[:self._size].std()
        self._rewards[:self._size] -= self.reward_mean
        self._rewards[:self._size] /= self.reward_std


class DelayBuffer:
    def __init__(self, observation_dim, action_dim, delay=5, buffer_size=int(1e6), device="cpu"):

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._delay = delay
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._observations = torch.zeros((buffer_size, delay + 1, observation_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, delay + 1, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, delay + 1, 1), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, delay + 1, 1), dtype=torch.float32, device=device)
        self._masks = torch.zeros((buffer_size, delay + 1, 1), dtype=torch.float32, device=device)
        self._device = device

        self._padding_observations = torch.zeros((delay + 1, observation_dim), dtype=torch.float32, device=device)
        self._padding_actions = torch.zeros((delay + 1, action_dim), dtype=torch.float32, device=device)
        self._padding_rewards = torch.zeros((delay + 1, 1), dtype=torch.float32, device=device)
        self._padding_dones = torch.zeros((delay + 1, 1), dtype=torch.float32, device=device)

    def _to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, dataset_name):
        env = gym.make(dataset_name)
        dataset = env.get_dataset()
        n_transitions = dataset["observations"].shape[0]
        delay_seq = {
            'observations': deque(maxlen=self._delay+1),
            'actions': deque(maxlen=self._delay+1),
            'rewards': deque(maxlen=self._delay+1),
            'dones': deque(maxlen=self._delay+1),
        }
        for i in trange(n_transitions):
        # for i in trange(50000):
            delay_seq["observations"].append(dataset["observations"][i])
            delay_seq["actions"].append(dataset["actions"][i])
            delay_seq["rewards"].append(dataset["rewards"][i])
            delay_seq["dones"].append(np.logical_or(dataset["terminals"][i], dataset["timeouts"][i]))
            if len(delay_seq['observations']) != self._delay + 1:
                # continue no padding
                # print('no padding')
                padding_length = self._delay+1 - len(delay_seq['observations'])
                self._observations[self._size] = torch.cat(
                    (self._to_tensor(np.array(list(delay_seq["observations"]))),
                     self._padding_observations[:padding_length]), dim=0)
                self._actions[self._size] = torch.cat(
                    (self._to_tensor(np.array(list(delay_seq["actions"]))),
                     self._padding_actions[:padding_length]), dim=0)
                self._rewards[self._size] = torch.cat(
                    (self._to_tensor(np.array(list(delay_seq["rewards"]))).unsqueeze(-1),
                     self._padding_rewards[:padding_length]), dim=0)
                self._dones[self._size] = torch.cat(
                    (self._to_tensor(np.array(list(delay_seq["dones"]))).unsqueeze(-1),
                     self._padding_dones[:padding_length]), dim=0)
                self._masks[self._size][padding_length:] = 1
            else:
                self._observations[self._size] = self._to_tensor(np.array(list(delay_seq["observations"])))
                self._actions[self._size] = self._to_tensor(np.array(list(delay_seq["actions"])))
                self._rewards[self._size] = self._to_tensor(np.array(list(delay_seq["rewards"]))).unsqueeze(-1)
                self._dones[self._size] = self._to_tensor(np.array(list(delay_seq["dones"]))).unsqueeze(-1)
            self._pointer += 1
            self._size += 1
        # print(f'loaded {dataset_name}, n_transitions {n_transitions}')
        print(f'loaded {dataset_name}, n_transitions {self._size}')

    def generate_sample_prior(self, batch_size=256):
        sample_prior = np.arange(self._size)
        np.random.shuffle(sample_prior)
        self._sample_prior = np.array_split(
            sample_prior, 
            self._size // batch_size
        )
        # print(f'generating sample prior {len(self._sample_prior)}')
        return self._sample_prior

    def sample(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.random.randint(0, self._size, size=batch_size)
        observations = self._observations[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        dones = self._dones[indices]
        masks = self._masks[indices]
        return [observations, actions, rewards, dones, masks]

    def normalize_reward(self):
        self.reward_mean = self._rewards[:self._size].mean()
        self.reward_std = self._rewards[:self._size].std()
        self._rewards[:self._size] -= self.reward_mean
        self._rewards[:self._size] /= self.reward_std


class GaussianNormalizer:
    """ Gaussian Normalizer

    Normalizes data to have zero mean and unit variance.
    For those dimensions with zero variance, the normalized value will be zero.

    Args:
        X: np.ndarray,
            dataset with shape (..., *x_shape)
        start_dim: int,
            the dimension to start normalization from, Default: -1

    Examples:
        >>> x_dataset = np.random.randn(100000, 3, 10)

        >>> normalizer = GaussianNormalizer(x_dataset, 1)
        >>> x = np.random.randn(1, 3, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)

        >>> normalizer = GaussianNormalizer(x_dataset, 2)
        >>> x = np.random.randn(1, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)
    """

    def __init__(self, X: np.ndarray, start_dim: int = -1):
        total_dims = X.ndim
        if start_dim < 0:
            start_dim = total_dims + start_dim

        axes = tuple(range(start_dim))

        self.mean = np.mean(X, axis=axes)
        self.std = np.std(X, axis=axes)
        self.std[self.std == 0] = 1.

    def normalize(self, x: np.ndarray):
        ndim = x.ndim
        return (x - at_least_ndim(self.mean, ndim, 1)) / at_least_ndim(self.std, ndim, 1)

    def unnormalize(self, x: np.ndarray):
        ndim = x.ndim
        return x * at_least_ndim(self.std, ndim, 1) + at_least_ndim(self.mean, ndim, 1)



class D4RLMuJoCoDataset(Dataset):
    """ **D4RL-MuJoCo Sequential Dataset**

    torch.utils.data.Dataset wrapper for D4RL-MuJoCo dataset.
    Chunk the dataset into sequences of length `horizon` without padding.
    Use GaussianNormalizer to normalize the observations as default.
    Each batch contains
    - batch["obs"]["state"], observations of shape (batch_size, horizon, o_dim)
    - batch["act"], actions of shape (batch_size, horizon, a_dim)
    - batch["rew"], rewards of shape (batch_size, horizon, 1)
    - batch["val"], Monte Carlo return of shape (batch_size, 1)

    Args:
        dataset: Dict[str, np.ndarray],
            D4RL-MuJoCo dataset. Obtained by calling `env.get_dataset()`.
        terminal_penalty: float,
            Penalty reward for early-terminal states. Default is -100.
        horizon: int,
            Length of each sequence. Default is 1.
        max_path_length: int,
            Maximum length of the episodes. Default is 1000.
        discount: float,
            Discount factor. Default is 0.99.

    Examples:
        >>> env = gym.make("halfcheetah-medium-expert-v2")
        >>> dataset = D4RLMuJoCoDataset(env.get_dataset(), horizon=32)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> obs = batch["obs"]["state"]  # (32, 32, 17)
        >>> act = batch["act"]           # (32, 32, 6)
        >>> rew = batch["rew"]           # (32, 32, 1)
        >>> val = batch["val"]           # (32, 1)

        >>> normalizer = dataset.get_normalizer()
        >>> obs = env.reset()[None, :]
        >>> normed_obs = normalizer.normalize(obs)
        >>> unnormed_obs = normalizer.unnormalize(normed_obs)
    """
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            terminal_penalty: float = -100.,
            horizon: int = 1,
            max_path_length: int = 1000,
            discount: float = 0.99,
    ):
        super().__init__()

        observations, actions, rewards, timeouts, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["timeouts"],
            dataset["terminals"])
        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        n_paths = np.sum(np.logical_or(terminals, timeouts))
        self.seq_obs = np.zeros((n_paths, max_path_length, self.o_dim), dtype=np.float32)
        self.seq_act = np.zeros((n_paths, max_path_length, self.a_dim), dtype=np.float32)
        self.seq_rew = np.zeros((n_paths, max_path_length, 1), dtype=np.float32)
        self.seq_val = np.zeros((n_paths, max_path_length, 1), dtype=np.float32)
        self.tml_and_not_timeout = []
        self.indices = []

        path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i]:
                path_lengths.append(i - ptr + 1)

                if terminals[i] and not timeouts[i]:
                    rewards[i] = terminal_penalty if terminal_penalty is not None else rewards[i]
                    self.tml_and_not_timeout.append([path_idx, i - ptr])

                self.seq_obs[path_idx, :i - ptr + 1] = normed_observations[ptr:i + 1]
                self.seq_act[path_idx, :i - ptr + 1] = actions[ptr:i + 1]
                self.seq_rew[path_idx, :i - ptr + 1] = rewards[ptr:i + 1][:, None]

                max_start = min(path_lengths[-1] - 1, max_path_length - horizon)
                self.indices += [(path_idx, start, start + horizon) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1

        self.seq_val[:, -1] = self.seq_rew[:, -1]
        for i in range(max_path_length - 1):
            self.seq_val[:, - 2 - i] = self.seq_rew[:, -2 - i] + discount * self.seq_val[:, -1 - i]
        self.path_lengths = np.array(path_lengths)
        self.tml_and_not_timeout = np.array(self.tml_and_not_timeout, dtype=np.int64)

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        data = {
            'obs': {
                'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
            'rew': self.seq_rew[path_idx, start:end],
            'val': self.seq_val[path_idx, start],
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data
