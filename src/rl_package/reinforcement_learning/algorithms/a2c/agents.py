import torch
import numba
import gymnasium
import random
from typing import Optional, Callable
from rl_package.reinforcement_learning.algorithms.base_classes import *

class A2C(MultiAgentRLPipeline):

    """
    Implementation of the A2C algorithm. Uses parallel agents

    acting in parallel environments to leverage different

    exploration policies.

    Original Paper: Asynchronous Methods for Deep Reinforcement

    Learning by Mnih et al, https://arxiv.org/pdf/1602.01783v2.pdf

    Fields:

        `gymnasium.vector.VectorEnv` environment: the environment to train in.

        `torch.nn.Module` actor_network: the actor network.

        `torch.nn.Module` critic_network: the critic network.

        `bool` discrete: whether the pipeline is made for discrete or continuous action spaces.
    """

    actor_network: torch.nn.Module 

    critic_network: torch.nn.Module 

    discrete: bool

    device: torch.device

    def __init__(self, 
        environment: gymnasium.Env, 
        num_threads: int,
        discrete: bool,
        actor_network_architecture: torch.nn.Module, 
        critic_network_architecture: torch.nn.Module,
        device: torch.device,
    ):
        
        action_space: gymnasium.Space = environment.action_space

        observation_space: gymnasium.Space = environment.observation_space

        vectorized_environment: gymnasium.vector.VectorEnv = gymnasium.vector.VectorEnv(
            num_threads, 
            observation_space,
            action_space
        )

        super().__init__(vectorized_environment)

        self.actor_network = actor_network_architecture.to(device)

        self.critic_network = critic_network_architecture.to(device)

        self.discrete = discrete

        self.device = device

    def epsilon_greedy_action(self, state: ArrayType, epsilon: float) -> int:

        random_number: float = random.random()

        if random_number >= epsilon:

            return self.environment.action_space.sample()

        else:

            with torch.no_grad():

                pass