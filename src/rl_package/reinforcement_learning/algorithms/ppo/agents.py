import torch
import gymnasium
from typing import Callable, Optional
from rl_package.reinforcement_learning.algorithms.base_classes import *
from rl_package.reinforcement_learning.algorithms.ppo.utils import *
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

    def action(self, state: ArrayType, **_: dict[str, Any]) -> int:

        with torch.no_grad():

            log_action_distribution: torch.Tensor = torch.nn.functional.log_softmax(self.actor(state.reshape((1,)+state.shape)))

            return torch.argmax(log_action_distribution).item()
        
    def train(self, discount_factor: float = 0.99, state_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, **kwargs: dict[str, torch.Any]) -> None:
        
        clip_epsilon: Callable[[int], float] = kwargs["epsilon"]

        # important note; if task is episodic, "timesteps" must be much less than episode length
        timesteps: int = kwargs["timesteps"]

        iterations: torch.Tensor = kwargs["iterations"]

        epochs: int = kwargs['epochs']

        batch_size: int = kwargs["batch_size"]

        states: list[torch.Tensor] = []

        actions: list[int] = []

        rewards: list[float] = []

        current_state: torch.Tensor = torch.as_tensor(self.environment.reset()[0]).to(self.device)

        terminated: bool = False

        truncated: bool = False

        for iteration in range(iterations):

            timestep: int = 0

            if state_transform:

                current_state = state_transform(current_state)

            states.append(current_state)

            while timestep < timesteps and not terminated and not truncated:

                action: ActionType = self.action(current_state)

                actions.append(action)

                next_state, reward, terminated, truncated, _ = self.environment.step(action)

                rewards.append(reward)

                next_state = torch.as_tensor(next_state).to(self.device)

                if state_transform:

                    next_state = state_transform(next_state)

                current_state = next_state

                states.append(current_state)

                timestep += 1

            if terminated or truncated:

                terminated = False

                truncated = False

                current_state: torch.Tensor = torch.as_tensor(self.environment.reset()[0]).to(self.device)

        states_tensor: torch.Tensor = torch.stack(states, dim=0)

        rewards_tensor: torch.Tensor = torch.stack(rewards, dim=0)

        advantage_estimates: torch.Tensor = self._compute_advantage_estimates(states_tensor, rewards_tensor, discount_factor)

        advantage_dataset: data.Dataset = AdvantageEstimateDataset(advantage_estimates)

        advantage_dataloader: data.DataLoader = data.DataLoader(advantage_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):

            for advantage_batch in advantage_dataloader:

                probability_ratio: float = self._probability_ratio()
    
    def _probability_ratio(self, state: torch.Tensor, action: int) -> float:

        """
        Calculates the probability ratio between the old (current)

        policy and the current updates policy.

        Arguments:

            `torch.Tensor` state: the state for the probability ratio.

            `int` action: the action for the probability ratio.
        """

        old_policy_probability: torch.Tensor = self.actor_old(state)[action]

        new_policy_probability: torch.Tensor = self.actor(state)[action]

        return old_policy_probability / new_policy_probability
        
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