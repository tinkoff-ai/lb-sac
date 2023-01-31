import math
import torch
import torch.nn as nn

from torch.distributions import Normal
from typing import Tuple, Optional


class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 3, "shape should be [num_models, batch_size, in_features]"
        return torch.bmm(input, self.weight) + self.bias

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ensemble_size={self.ensemble_size}'


class Actor(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int,
            max_action: float = 1.0,
            layernorm: bool = False,
            edac_init: bool = False
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.ReLU(),
        )
        # works better in practice with separate layers
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        if edac_init:
            for layer in self.trunk[::3]:
                torch.nn.init.constant_(layer.bias, 0.1)

            torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
            self,
            state: torch.Tensor,
            greedy: bool = False,
            need_logprob: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if greedy:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_logprob:
            # change of variable formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob


class EnsembleCritic(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int,
            num_critics: int,
            layernorm: bool = False,
            edac_init: bool = False
    ):
        super().__init__()
        self.ensemble = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics)
        )
        if edac_init:
            for layer in self.ensemble[::3]:
                torch.nn.init.constant_(layer.bias, 0.1)

            torch.nn.init.uniform_(self.ensemble[-1].weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(self.ensemble[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        out = self.ensemble(state_action).squeeze(-1)
        return out