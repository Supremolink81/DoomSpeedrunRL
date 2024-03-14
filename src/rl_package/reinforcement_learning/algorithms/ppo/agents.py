import torch
import numba
import gymnasium
import random
from ..base_classes import *

@numba.jitclass([
    ("environment", gymnasium.Env),
    ("agent", list[torch.nn.Module]),
])
class PPO(RLPipeline):

    """
    Implementation of Proximal Policy Optimization, a policy

    gradient optimization algorithm that uses advantage estimates,

    multiple agents, and a bias-variance regularizer.

    Original Paper: https://arxiv.org/pdf/1707.06347.pdf

    Fields:

        `gymnasium.Env` environment: the environment the agent resides in.

        `list[torch.nn.Module]` agents: the agents to train.
    """

    actor: torch.nn.Module

    def __init__(self, environment: gymnasium.Env, actor_architecture: torch.nn.Module):

        super().__init__(environment)

        self.actor = actor_architecture