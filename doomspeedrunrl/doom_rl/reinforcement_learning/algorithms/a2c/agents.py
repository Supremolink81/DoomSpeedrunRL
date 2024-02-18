import torch
import numba
import gymnasium
import random
from typing import Optional, Callable
from base_classes import *

class A2C(MultiAgentRLPipeline):

    """
    Implementation of the A2C algorithm. Uses parallel agents

    acting in parallel environments to leverage different

    exploration policies.

    Original Paper: Asynchronous Methods for Deep Reinforcement

    Learning by Mnih et al, https://arxiv.org/pdf/1602.01783v2.pdf

    Fields:

        `gymnasium.vector.VectorEnv` environment: the environment to train in.

        `torch.nn.Module` target_q_function: the target Q function.
    """

    target_q_function: torch.nn.Module

    @numba.jit(nopython=True)
    def __init__(self, environment: gymnasium.vector.VectorEnv, target_q_function_architecture: torch.nn.Module):

        super().__init__(environment)

        self.target_q_function = target_q_function_architecture

    @numba.jit(nopython=True)
    def epsilon_greedy_action(self, state: ArrayType, epsilon: float) -> int:

        random_number: float = random.random()

        if random_number >= epsilon:

            return self.environment.action_space.sample()

        else:

            with torch.no_grad():

                action_distribution: torch.Tensor = self.target_q_function(state.reshape((1,)+state.shape))

                return torch.argmax(action_distribution)[0]