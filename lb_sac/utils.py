import os
import gym
import d4rl
import torch
import random
import collections
import numpy as np
import torch.nn as nn

from typing import Optional


# source: https://github.com/rail-berkeley/d4rl/blob/d842aa194b416e564e54b0730d9f934e3e32f854/d4rl/__init__.py#L63
# modified to also return next_action (needed for SARSA training)
def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, next_actions, rewards,
     and a terminal flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            next_actions: An N x dim_action array of next actions.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        new_action = dataset['actions'][i + 1].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'next_actions': np.array(next_action_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


# source: https://github.com/rail-berkeley/d4rl/blob/d842aa194b416e564e54b0730d9f934e3e32f854/d4rl/__init__.py#L137
# fixed bugs, also return next_obs & timeouts
def sequence_dataset(env, dataset=None, **kwargs):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            next_observations
            terminals
            timeouts
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatibility.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])
        data_["next_observations"].append(dataset["next_observations"][i])
        data_["terminals"].append(dataset["terminals"][i])
        data_["timeouts"].append(dataset["timeouts"][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


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


def max_grad_norm(module: nn.Module) -> float:
    norms = [p.grad.data.norm(2).item() for p in module.parameters() if p.grad is not None]
    return max(norms)


def soft_update(target: nn.Module, source: nn.Module, tau: float = 1e-3):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


class OfflineReplayBuffer:
    def __init__(
            self,
            dataset_name: str,
            need_next_action: bool = False,
            normalize_reward: bool = False,
            normalize_state: bool = False,
            device: str = "cpu"
    ):
        data = qlearning_dataset(gym.make(dataset_name))
        self.states = torch.tensor(data["observations"], dtype=torch.float32, device=device)
        self.actions = torch.tensor(data["actions"], dtype=torch.float32, device=device)
        self.rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=device)
        self.next_states = torch.tensor(data["next_observations"], dtype=torch.float32, device=device)
        self.dones = torch.tensor(data["terminals"], dtype=torch.float32, device=device)
        if need_next_action:
            self.next_actions = torch.tensor(data["next_actions"], dtype=torch.float32, device=device)

        self.dataset_name = dataset_name
        self.need_next_action = need_next_action
        self.buffer_size = len(self.states)

        if normalize_reward:
            self.__normalize_reward()
        
        if normalize_state:
            self.__normalize_state()
        
        del data

    def get_moments(self, modality: str):
        if modality == "state":
            mean = self.states.mean(0)
            std = self.states.std(0)
        elif modality == "action":
            mean = self.actions.mean(0)
            std = self.actions.std(0)
        elif modality == "reward":
            mean = self.rewards.mean(0)
            std = self.rewards.std(0)
        else:
            raise RuntimeError("Unknown modality! Should be one of: [state, action, reward]")
        return mean, std

    def __normalize_state(self):
        mean, std = self.get_moments("state")        
        self.states = (self.states - mean) / (std + 1e-8)
        self.next_states = (self.next_states - mean) / (std + 1e-8)
        print("Normalizing states.")

    def __normalize_reward(self):
        old_mean = self.rewards.mean()
        old_min = self.rewards.min()
        old_max = self.rewards.max()
        # normalization like in IQL:
        # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/train_offline.py#L35
        if any(s in self.dataset_name for s in ("halfcheetah", "hopper", "walker2d")):
            trajectories = list(sequence_dataset(gym.make(self.dataset_name)))
            trajectories.sort(key=lambda d: d["rewards"].sum())

            self.rewards /= trajectories[-1]["rewards"].sum() - trajectories[0]["rewards"].sum()
            self.rewards *= 1000.0
        elif "antmaze" in self.dataset_name:
            self.rewards -= 1
        else:
            raise RuntimeError("Can't normalize this dataset!")

        print(f"Normalizing rewards. Mean: {old_mean} -> {self.rewards.mean()},"
              f" Max: {old_max} -> {self.rewards.max()},"
              f" Min: {old_min} -> {self.rewards.min()}")

    def random_batch(self, batch_size):
        idxs = np.random.randint(self.buffer_size, size=batch_size)
        if self.need_next_action:
            batch = (
                self.states[idxs],
                self.actions[idxs],
                self.rewards[idxs],
                self.next_states[idxs],
                self.next_actions[idxs],
                self.dones[idxs]
            )
        else:
            batch = (
                self.states[idxs],
                self.actions[idxs],
                self.rewards[idxs],
                self.next_states[idxs],
                self.dones[idxs]
            )
        return batch


def eval_rollout(env: gym.Env, model) -> float:
    total_reward, total_steps = 0, 0
    state, done = env.reset(), False
    while not done:
        state, reward, done, _ = env.step(model.act(state, greedy=True))
        total_reward += reward
        total_steps += 1.0

    return total_reward
