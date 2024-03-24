import torch
import gymnasium
from rl_package.reinforcement_learning.algorithms.base_classes import *
from rl_package.utils.arrays import triangular_power_matrix

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

    device: torch.device

    def __init__(self, environment: gymnasium.Env, actor_architecture: torch.nn.Module, device: torch.device):

        super().__init__(environment)

        self.actor = actor_architecture.to(device)

        self.actor_old = actor_architecture.to(device)

        self.device = device
    
    def _probability_ratio(self, state: torch.Tensor, action: int, alpha: float) -> float:

        """
        Calculates the probability ratio between the old (current)

        policy and a new policy which is a convex combination of the

        old policy and the greedy policy
        """

        old_policy_probability: torch.Tensor = self.actor_old(state)[action]

        new_policy_probability: torch.Tensor = self.actor(state)[action]

        return old_policy_probability / new_policy_probability

    def action(self, state: ArrayType, **kwargs: dict[str, Any]) -> int:

        with torch.no_grad():

            log_action_distribution: torch.Tensor = torch.nn.functional.log_softmax(self.actor(state.reshape((1,)+state.shape)))

            return torch.argmax(log_action_distribution).item()
        
    def _state_value_function(self, states: torch.Tensor) -> torch.Tensor:

        """
        Uses the state-action value function and a discrete softmax policy

        to calculate the state value function.

        Arguments:

            `torch.Tensor` states: the environment state.

        Returns:

            a torch.Tensor with a value representing the state value function's output.
        """

        action_distribution: torch.Tensor = self.actor(states)

        action_values: torch.Tensor = self.critic(states)

        return torch.sum(action_values * action_distribution, axis=0)
    
    def _compute_advantage_estimates(self, states: torch.Tensor, rewards: torch.Tensor, discount_factor: float) -> torch.Tensor:

        """
        Computes the advantage estimates over a set of timesteps using

        a sequence of states, a sequence of rewards and a state value function.

        Arguments:

            `torch.Tensor` states: the sequence of states.

            `torch.Tensor` rewards: the sequence of rewards.

            `float` discount_factor: the discount factor. 

        Returns:

            the sequence of len(states) advantage estimates.
        """

        timesteps: int = states.shape[0]

        discount_factor_matrix: torch.Tensor = triangular_power_matrix(timesteps, discount_factor).to(self.device)

        # has values [discount_factor**(T-1), discount_factor**(T-2), ..., discount_factor, 1]
        discount_factor_vector: torch.Tensor = discount_factor ** (timesteps - torch.arange(1, timesteps+1))

        last_state_value: torch.Tensor = self._state_value_function(states[timesteps-1])

        discounted_last_state_values: torch.Tensor = last_state_value * discount_factor_vector

        target_returns: torch.Tensor = discount_factor_matrix @ rewards + discounted_last_state_values

        state_values: torch.Tensor = self._state_value_function(states)

        return target_returns - state_values