import os
import uuid
import wandb
import pyrallis

import gym
from typing import Optional
from dataclasses import dataclass, asdict

from lb_sac.sac import SACN
from lb_sac.networks import EnsembleCritic, Actor
from lb_sac.training import train_offline
from lb_sac.utils import OfflineReplayBuffer, set_seed


@dataclass
class TrainConfig:
    # wandb params
    project: str = "SAC-N"
    group: str = "sac-n-default"
    name: str = "sac-n-default"
    # model params
    hidden_dim: int = 256
    num_critics: int = 2
    gamma: float = 0.99
    tau: float = 5e-3
    eta: float = -1.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    actor_use_layernorm: bool = False
    critic_use_layernorm: bool = False
    edac_init: bool = False
    edac_style_critic_loss: bool = False
    max_action: float = 1.0
    actor_update_every: int = 1
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 1000
    num_updates_on_epoch: int = 1000
    checkpoints_path: Optional[str] = None
    save_every: int = 100
    normalize_reward: bool = False
    normalize_state: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 5
    # general params
    train_seed: int = 10
    eval_seed: int = 42
    log_every: int = 100
    device: str = "cuda"

    def __post_init__(self):
        tmp_env = gym.make(self.dataset_name)
        self.state_dim = tmp_env.observation_space.shape[0]
        self.action_dim = tmp_env.action_space.shape[0]

        self.name = f"{self.name}-{self.dataset_name}-seed{str(self.train_seed)}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed)
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        id=str(uuid.uuid4())  # for mlc, manually change it every init
    )
    buffer = OfflineReplayBuffer(
        dataset_name=config.dataset_name,
        need_next_action=True,
        normalize_reward=config.normalize_reward,
        normalize_state=config.normalize_state,
        device=config.device
    )

    actor = Actor(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        max_action=config.max_action,
        layernorm=config.actor_use_layernorm,
        edac_init=config.edac_init
    ).to(config.device)

    critic = EnsembleCritic(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        num_critics=config.num_critics,
        layernorm=config.critic_use_layernorm,
        edac_init=config.edac_init
    ).to(config.device)

    sac = SACN(
        actor=actor,
        critic=critic,
        gamma=config.gamma,
        tau=config.tau,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        alpha_lr=config.alpha_lr,
        actor_update_every=config.actor_update_every,
        eta=config.eta,
        edac_style_critic_loss=config.edac_style_critic_loss
    )
    train_offline(
        agent=sac,
        buffer=buffer,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        num_updates_on_epoch=config.num_updates_on_epoch,
        eval_env=gym.make(config.dataset_name),
        eval_episodes=config.eval_episodes,
        eval_every=config.eval_every,
        log_every=config.log_every,
        save_every=config.save_every,
        checkpoints_path=config.checkpoints_path,
        train_seed=config.train_seed,
        eval_seed=config.eval_seed,
        device=config.device
    )
    run.finish()


if __name__ == '__main__':
    train()
