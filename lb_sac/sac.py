import torch

import numpy as np
from typing import Tuple, List, Dict, Any
from copy import deepcopy
from torch.optim import Adam

from lb_sac.utils import soft_update, max_grad_norm
from lb_sac.networks import Actor, EnsembleCritic


class SACN:
    def __init__(
            self,
            actor: Actor,
            critic: EnsembleCritic,
            gamma: float = 0.99,
            tau: float = 0.005,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            alpha_lr: float = 1e-4,
            actor_update_every: int = 1,
            eta: float = -1.0,
            edac_style_critic_loss: bool = False
    ):
        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.tau = tau
        self.gamma = gamma
        self.actor_update_every = actor_update_every
        self.eta = eta
        self.edac_style_critic_loss = edac_style_critic_loss
        # will work only for 1-gpu training, tmp
        self.device = next(self.critic.parameters()).device

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)

        self.alpha_optim = Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.__updates_count = 0.0

    def _alpha_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        with torch.no_grad():
            action, action_log_prob = self.actor(state, need_logprob=True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss, {}

    def _actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        action, action_log_prob = self.actor(state, need_logprob=True)

        Q_value_dist = self.critic(state, action)
        assert Q_value_dist.shape[0] == self.critic.num_critics
        Q_value_min = Q_value_dist.min(0).values
        Q_value_std = Q_value_dist.std(0).mean().item()  # needed for logging

        assert action_log_prob.shape == Q_value_min.shape
        loss = (self.alpha * action_log_prob - Q_value_min).mean()

        loss_info = {
            "batch_entropy": -action_log_prob.mean().item(),
            "q_policy_std": Q_value_std,
        }
        return loss, loss_info

    def _critic_diversity_loss(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
    ) -> torch.Tensor:
        # almost exact copy from the original implementation, but with some style changes, source:
        # https://github.com/snu-mllab/EDAC/blob/198d5708701b531fd97a918a33152e1914ea14d7/lifelong_rl/trainers/q_learning/sac.py#L192
        state = state.unsqueeze(0).repeat_interleave(self.critic.num_critics, dim=0)
        action = action.unsqueeze(0).repeat_interleave(self.critic.num_critics, dim=0).requires_grad_(True)
        # [num_critics, batch_size]
        Q_dist = self.critic(state, action)

        Q_action_grad = torch.autograd.grad(Q_dist.sum(), action, retain_graph=True, create_graph=True)[0]
        Q_action_grad = Q_action_grad / (torch.norm(Q_action_grad, p=2, dim=2).unsqueeze(-1) + 1e-10)
        Q_action_grad = Q_action_grad.transpose(0, 1)

        # removed einsum as it is usually slower than just torch.bmm
        Q_action_grad = Q_action_grad @ Q_action_grad.permute(0, 2, 1)

        masks = torch.eye(self.critic.num_critics, device=self.device).unsqueeze(0).repeat(Q_action_grad.shape[0], 1, 1)
        Q_action_grad = (1 - masks) * Q_action_grad

        grad_loss = torch.mean(torch.sum(Q_action_grad, dim=(1, 2))) / (self.critic.num_critics - 1)

        return grad_loss

    def _critic_loss(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            next_state: torch.Tensor,
            next_action_data: torch.Tensor,
            done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        loss_info = {}
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, need_logprob=True)
            Q_value_dist = self.target_critic(next_state, next_action)

            loss_info["action_mse_mean"] = torch.mean((next_action - next_action_data)**2)
            loss_info["q_target_min"] = Q_value_dist.min(0).values.mean().item()

            Q_next = Q_value_dist.min(0).values
            Q_next = Q_next - self.alpha * next_action_log_prob
            Q_target = reward + self.gamma * (1 - done) * Q_next
            assert Q_next.shape == next_action_log_prob.shape == reward.shape

        Q_values = self.critic(state, action)
        # [num_critics, batch_size] - [1, batch_size]
        error = Q_values - Q_target.unsqueeze(0)
        if self.edac_style_critic_loss:
            loss = (error ** 2).mean(dim=1).sum(dim=0)
        else:
            loss = torch.mean(error ** 2)

        loss_info["q_error_mean"] = torch.mean(error).item()
        loss_info["q_data_std"] = Q_values.std(0).mean().item()

        if self.eta > 0:
            diversity_loss = self._critic_diversity_loss(state, action)
            loss = loss + self.eta * diversity_loss

        return loss, loss_info

    def update(self, batch: List[torch.Tensor]) -> Dict[str, Any]:
        state, action, reward, next_state, next_action, done = batch

        update_info = {}
        # Alpha update
        alpha_loss, alpha_info = self._alpha_loss(state)
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        update_info.update(alpha_info)

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        if self.__updates_count % self.actor_update_every == 0:
            actor_loss, actor_info = self._actor_loss(state)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            update_info.update(actor_info)
            update_info["actor_grad_max_norm"] = max_grad_norm(self.actor)
            update_info["actor_loss"] = actor_loss.item()

        # Critic update
        critic_loss, critic_info = self._critic_loss(state, action, reward, next_state, next_action, done)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        update_info.update(critic_info)
        update_info["critic_grad_max_norm"] = max_grad_norm(self.critic)

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)

            # for logging, std estimate of the random actions ~ U[-max_action, max_action]
            random_actions = (
                    -self.actor.max_action + 2 * self.actor.max_action * torch.rand_like(action)
            )
            Q_random_dist = self.critic(state, random_actions)
            update_info["q_random_std"] = Q_random_dist.std(0).mean().item()

        self.__updates_count += 1
        update_info.update({
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha": self.alpha.item(),
        })
        return update_info

    @torch.no_grad()
    def act(self, state: np.ndarray, greedy: bool = False) -> np.ndarray:
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action = self.actor(state, greedy=greedy)[0].cpu().numpy()
        return action

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict()
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.alpha_optim.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()