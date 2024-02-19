import torch
import gymnasium
from doom_rl.reinforcement_learning.algorithms.base_classes import *
from doom_rl.reinforcement_learning.algorithms.ddpg.loss import *
from typing import Callable, Optional

class DDPG(SingleAgentRLPipeline):

    """
    Implementation of the Deep Deterministic Policy Gradient

    Algorithm for continuous action spaces. Uses an actor network

    to choose actions, and a critic network to update the optimal

    paths and the actor. 

    Original Paper: Continuous Control With Deep

    Reinforcement Learning by Lillicrap et al: https://arxiv.org/pdf/1509.02971.pdf

    Fields:

        `gymnasium.Env` environment: the environment the agent resides in.

        `torch.nn.Module` critic_network: the critic network for the actor.

        `torch.nn.Module` target_critic_network: the target critic network for updates.

        `torch.nn.Module` actor_network: the actor network that acts as the policy.

        `torch.nn.Module` target_actor_network: the target actor network for updates.
    """

    critic_network: torch.nn.Module

    target_critic_network: torch.nn.Module

    actor_network: torch.nn.Module

    target_actor_network: torch.nn.Module

    def __init__(self, 
        environment: gymnasium.Env, 
        critic_network_architecture: torch.nn.Module, 
        actor_network_architecture: torch.nn.Module, 
        device: torch.device
    ):

        super().__init__(environment)

        self.critic_network = critic_network_architecture.to(device)
        
        self.target_critic_network = critic_network_architecture.to(device)

        self.target_critic_network.load_state_dict(self.critic_network.state_dict())

        self.actor_network = actor_network_architecture.to(device)

        self.target_actor_network = actor_network_architecture.to(device)

        self.target_actor_network.load_state_dict(self.actor_network.state_dict())
        
        self.device = device

    def epsilon_greedy_action(self, state: ArrayType, epsilon: float) -> int:

        random_number: float = random.random()

        if random_number < epsilon:

            return self.environment.action_space.sample()

        else:

            with torch.no_grad():

                action_distribution: torch.Tensor = self.actor_network(state.reshape((1,)+state.shape))

                return action_distribution.cpu().numpy()
            
    def train(self, **kwargs: Dict[str, Any]) -> None:

        self.no_rendering()

        buffer: ReplayBuffer = ReplayBuffer(kwargs["replay_buffer_capacity"])

        actor_optimizer: torch.optim.Optimizer = kwargs["actor_optimizer"]

        critic_optimizer: torch.optim.Optimizer = kwargs["critic_optimizer"]

        episodes: int = kwargs["episodes"]

        discount_factor: float = kwargs["discount_factor"]

        epsilon: float = kwargs["epsilon"]

        batch_size: int = kwargs["batch_size"]

        state_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = kwargs.get("state_transform", None)

        for episode in range(episodes): 

            terminated: bool = False

            truncated: bool = False

            current_state: torch.Tensor = torch.as_tensor(self.environment.reset()[0]).to(self.device)

            if state_transform:

                current_state = state_transform(current_state)

            while not terminated and not truncated:

                action: ActionType = self.epsilon_greedy_action(current_state, epsilon)

                next_state, reward, terminated, truncated, _ = self.environment.step(action)

                next_state = torch.as_tensor(next_state).to(self.device)

                if state_transform:

                    next_state = state_transform(next_state)

                buffer.add((current_state, reward, action, next_state))

                critic_optimizer.zero_grad()

                batch: list[Transition] = buffer.sample(batch_size)

                critic_loss: torch.Tensor = loss_function_ddpg_critic(
                    self.critic_network,
                    self.target_critic_network,
                    self.target_actor_network,
                    batch,
                    discount_factor,
                    self.device
                )

                critic_loss.backward()

                critic_optimizer.step()

                batch: list[Transition] = buffer.sample(batch_size)

                actor_optimizer.zero_grad()

                critic_loss: torch.Tensor = loss_function_ddpg_actor(
                    self.critic_network,
                    batch,
                    self.device
                )

                critic_loss.backward()

                actor_optimizer.step()

                current_state = next_state.clone()

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