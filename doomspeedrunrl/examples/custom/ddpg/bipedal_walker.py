import torch
import gymnasium
from doom_rl.reinforcement_learning.algorithms.ddpg.agents import DDPG
from doom_rl.reinforcement_learning.architectures.continuous import BipedalWalkerActorMLP, BipedalWalkerCriticMLP

LEARNING_RATE = 0.0001

EPISODES = 1000

REPLAY_BUFFER_CAPACITY = 1000

DISCOUNT_FACTOR = 0.99

EPSILON = 0.25

BATCH_SIZE = 100

UPDATE_COEFFICIENT = 0.01

DEVICE = torch.device("cuda:0")

if __name__ == "__main__":

    environment: gymnasium.Env = gymnasium.make("BipedalWalker-v3", max_episode_steps=300)

    actor_architecture: BipedalWalkerActorMLP = BipedalWalkerActorMLP()

    critic_architecture: BipedalWalkerCriticMLP = BipedalWalkerCriticMLP()

    algorithm: DDPG = DDPG(environment, critic_architecture, actor_architecture, DEVICE)

    algorithm.train(
        critic_optimizer=torch.optim.SGD(critic_architecture.parameters(), lr=LEARNING_RATE),
        actor_optimizer=torch.optim.SGD(actor_architecture.parameters(), lr=LEARNING_RATE),
        episodes=EPISODES,
        replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=lambda episode: 1 / (1 + episode / 100),
        batch_size=BATCH_SIZE,
        update_coefficient=UPDATE_COEFFICIENT,
    )

    algorithm.run(episodes=2000)