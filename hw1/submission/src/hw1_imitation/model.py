"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from cmath import tau
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(state_dim, hidden_dims[0]))

        self.layers.append(nn.ReLU())
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dims[len(hidden_dims)-1], action_dim * chunk_size))

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action_model = self.sample_actions(state)
        return nn.functional.mse_loss(action_model, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        s = state
        for layer in self.layers:
            s = layer(s)
        return s.reshape(-1, self.chunk_size, self.action_dim)





class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        
        super().__init__(state_dim, action_dim, chunk_size)
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(state_dim + action_dim * chunk_size + 1, hidden_dims[0])) # + 1 for \tau time step

        self.layers.append(nn.ReLU())
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dims[len(hidden_dims)-1], action_dim * chunk_size))

    def run_model(self, state: torch.Tensor,
        action_chunk: torch.Tensor,
        tau: torch.Tensor) -> torch.Tensor:
        b = state.shape[0]
        action_chunk = action_chunk.reshape(b, -1)
        tau = torch.tensor(tau).expand(b, 1) 
        x = torch.cat([state, action_chunk, tau], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x.reshape(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        A_t_0 = torch.normal(0,1, size=action_chunk.shape)
        tau = torch.rand(1)
        A_t_tau = tau * action_chunk + (1-tau) * A_t_0
        flow_vector = self.run_model(state, A_t_tau, tau)
        return nn.functional.mse_loss(flow_vector, action_chunk - A_t_0)

    def sample_actions(
        self,
        state: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        b = state.shape[0]
        A_t_0 = torch.normal(0, 1, size=(b, self.chunk_size, self.action_dim))
        A_next = A_t_0
        tau = 0
        for i in range(num_steps):
            A_next = A_next + (self.run_model(state, A_next, tau) / num_steps)
            tau += 1/num_steps
        return A_next


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
