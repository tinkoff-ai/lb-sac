import os
import gym
import d4rl
import torch
import wandb
import numpy as np

from tqdm import trange
from typing import Optional
from lb_sac.utils import OfflineReplayBuffer, set_seed, eval_rollout


def train_offline(
        agent,
        buffer: OfflineReplayBuffer,
        batch_size: int = 256,
        num_epochs: int = 500,
        num_updates_on_epoch: int = 1000,
        eval_env: Optional[gym.Env] = None,
        eval_episodes: int = 10,
        log_every: int = 100,
        eval_every: int = 1,
        save_every: int = 1,
        checkpoints_path: Optional[str] = None,
        train_seed: int = 42,
        eval_seed: int = 10,
        device: str = "cpu"
):
    set_seed(train_seed)
    if checkpoints_path is not None:
        os.makedirs(checkpoints_path, exist_ok=True)

    total_updates = 0.0
    for epoch in trange(1, num_epochs + 1, desc="Training"):
        # training
        for _ in trange(num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.random_batch(batch_size)
            batch = [b.to(device) for b in batch]
            update_info = agent.update(batch)

            if total_updates % log_every == 0:
                wandb.log({"epoch": epoch, **update_info})
            total_updates += 1

        # evaluation
        if eval_env is not None and epoch % eval_every == 0:
            eval_env.seed(eval_seed)
            returns = [
                eval_rollout(eval_env, agent)
                for _ in trange(eval_episodes, desc="Evaluation", leave=False)
            ]
            normalized_scores = eval_env.get_normalized_score(np.array(returns)) * 100
            wandb.log({
                    "eval/reward_mean": np.mean(returns),
                    "eval/reward_std": np.std(returns),
                    "eval/normalized_score_mean": np.mean(normalized_scores),
                    "eval/normalized_score_std": np.std(normalized_scores),
                    "epoch": epoch,
            })

        # saving
        if checkpoints_path is not None and (epoch % save_every == 0 or epoch == num_epochs):
            torch.save(agent.state_dict(), os.path.join(checkpoints_path, f"{epoch}.pt"))
