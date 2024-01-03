import torch
import numba
import gymnasium
import random
from ..base_classes import *
from loss import loss_function

@numba.jitclass([
    ("environment", gymnasium.Env),
    ("q_function", torch.nn.Module),
])
class DQN(RLPipeline):

    q_function: torch.nn.Module

    def __init__(self, environment: gymnasium.Env, q_function_architecture: torch.nn.Module):

        super().__init__(environment)

        self.q_function = q_function_architecture
    
    def train(self, **kwargs: Dict[str, Any]) -> None:

        buffer: ReplayBuffer = ReplayBuffer(kwargs["replay_buffer_capacity"])

        optimizer: torch.optim.Optimizer = kwargs["optimizer"]

        episodes: int = kwargs["episodes"]

        discount_factor: float = kwargs["discount_factor"]

        epsilon: float = kwargs["epsilon"]

        batch_size: int = kwargs["batch_size"]

        for _ in range(episodes):

            terminated: bool = False

            truncated: bool = False

            current_state: torch.Tensor = torch.as_tensor(self.environment.reset()[0])

            while not terminated and not truncated:

                random_number: float = random.random()

                # this conditional implements the epsilon-greedy aspect of DQN.

                action: int

                if random_number >= epsilon:

                    action = self.environment.action_space.sample()

                else:

                    action_distribution: torch.Tensor = self.q_function(current_state.reshape((1,)+current_state.shape))

                    action = torch.argmax(action_distribution)[0]

                    del action_distribution

                next_state, reward, terminated, truncated = self.environment.step(action)

                non_terminal_state: int = not terminated and not truncated

                buffer.add((current_state, reward, action, next_state, non_terminal_state))

                optimizer.zero_grad()

                batch: list[Transition] = buffer.sample(batch_size)

                loss: torch.Tensor = loss_function(self.q_function, batch, discount_factor)

                loss.backward()

                optimizer.step()

            self.environment.reset()