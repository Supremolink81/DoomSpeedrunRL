import torch
import gymnasium
from rl_package.reinforcement_learning.algorithms.base_classes import *

class PPO(SingleAgentRLPipeline):

    """
    Implementation of Proximal Policy Optimization, a policy

    gradient optimization algorithm that uses advantage estimates,

    multiple agents, and a bias-variance regularizer.

    Original Paper: https://arxiv.org/pdf/1707.06347.pdf

    Fields:

        `gymnasium.Env` environment: the environment the agent resides in.

        `torch.nn.Module` actor: the actor to train.

        `torch.nn.Module` actor_old: placeholder member for the probability ratio.

        `torch.nn.Module` critic: the critic to train; represented as a state action value function.
    """

    actor_old: torch.nn.Module

    actor: torch.nn.Module

    critic: torch.nn.Module

    def __init__(self, environment: gymnasium.Env, actor_architecture: torch.nn.Module):

        super().__init__(environment)

        self.actor = actor_architecture

        self.actor_old = actor_architecture
    
    def _probability_ratio(self, state: torch.Tensor, action: int, alpha: float) -> float:

        """
        Calculates the probability ratio between the old (current)

        policy and a new policy which is a convex combination of the

        old policy and the greedy policy
        """

        old_policy_probability: torch.Tensor = self.actor_old(state)[action]

        new_policy_probability: torch.Tensor = self.actor(state)[action]

        return old_policy_probability / new_policy_probability

    def epsilon_greedy_action(self, state: ArrayType, epsilon: float) -> int:

        with torch.no_grad():

            log_action_distribution: torch.Tensor = torch.nn.functional.log_softmax(self.actor(state.reshape((1,)+state.shape)))

            return torch.argmax(log_action_distribution).item()