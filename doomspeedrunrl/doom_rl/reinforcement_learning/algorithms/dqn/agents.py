import torch
import gymnasium
import random
from typing import Optional, Callable
from doom_rl.reinforcement_learning.algorithms.base_classes import *
from doom_rl.reinforcement_learning.algorithms.dqn.loss import loss_function_dqn

class DQN(SingleAgentRLPipeline):

    """
    Implementation of Deep Q Learning. Uses the optimal action

    from the Q function in order to take actions.

    Original Paper: Playing Atari with Deep Reinforcement
     
    Learning by Mnih et al, https://arxiv.org/pdf/1312.5602.pdf

    Fields:

        `gymnasium.Env` environment: the environment the agent resides in.

        `torch.nn.Module` q_network: the Q function.
    """

    q_network: torch.nn.Module

    def __init__(self, environment: gymnasium.Env, q_network_architecture: torch.nn.Module, device: torch.device):

        super().__init__(environment)

        self.q_network = q_network_architecture.to(device)
        
        self.device = device

    def epsilon_greedy_action(self, state: ArrayType, epsilon: float) -> int:

        random_number: float = random.random()

        if random_number < epsilon:

            return self.environment.action_space.sample()

        else:

            with torch.no_grad():

                action_distribution: torch.Tensor = self.q_network(state.reshape((1,)+state.shape))

                return torch.argmax(action_distribution).item()
    
    def train(self, **kwargs: Dict[str, Any]) -> None:

        self.human_rendering()

        buffer: ReplayBuffer = ReplayBuffer(kwargs["replay_buffer_capacity"])

        optimizer: torch.optim.Optimizer = kwargs["optimizer"]

        episodes: int = kwargs["episodes"]

        discount_factor: float = kwargs["discount_factor"]

        epsilon: Callable[[int], float] = kwargs["epsilon"]

        batch_size: int = kwargs["batch_size"]

        state_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = kwargs.get("state_transform", None)

        for episode in range(episodes):

            terminated: bool = False

            truncated: bool = False

            current_state: torch.Tensor = torch.as_tensor(self.environment.reset()[0]).to(self.device)

            if state_transform:

                current_state = state_transform(current_state)

            while not terminated and not truncated:

                action: ActionType = self.epsilon_greedy_action(current_state, epsilon(episode))

                next_state, reward, terminated, truncated, _ = self.environment.step(action)

                next_state = torch.as_tensor(next_state).to(self.device)

                if state_transform:

                    next_state = state_transform(next_state)

                non_terminal_state: int = not terminated and not truncated

                buffer.add((current_state, reward, action, next_state, non_terminal_state))

                optimizer.zero_grad()

                batch: list[Transition] = buffer.sample(batch_size)

                loss: torch.Tensor = loss_function_dqn(self.q_network, batch, discount_factor,self.device)

                loss.backward()

                optimizer.step()

                current_state = next_state.clone()

                print(action, loss)

            print(f"Episode {episode+1} finished.")

    def run(self, episodes: int = -1, **kwargs: Dict[str, Any]) -> None:

        self.human_rendering()

        state_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = kwargs.get("state_transform", None)

        episode: int = 0

        with torch.no_grad():

            while episode < episodes:

                terminated: bool = False

                truncated: bool = False

                current_state: torch.Tensor = torch.as_tensor(self.environment.reset()[0]).to(self.device)

                if state_transform:

                    current_state = state_transform(current_state)

                while not terminated and not truncated:
                    
                    # epsilon = 0 since we don't want random actions
                    action: int = self.epsilon_greedy_action(current_state, 0)

                    next_state, _, terminated, truncated, _ = self.environment.step(action)

                    next_state = torch.as_tensor(next_state).to(self.device)

                    current_state = next_state.clone()

                self.environment.render()