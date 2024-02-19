import torch
import gymnasium
from doom_rl.reinforcement_learning.algorithms.dqn.agents import DQN
from doom_rl.reinforcement_learning.architectures.discrete import CartPoleMLP

LEARNING_RATE = 0.0000000001

EPISODES = 10000

REPLAY_BUFFER_CAPACITY = 10000

DISCOUNT_FACTOR = 0.99

BATCH_SIZE = 1000

DEVICE = torch.device("cuda:0")

if __name__ == "__main__":

    environment: gymnasium.Env = gymnasium.make("CartPole-v1")

    network_architecture: CartPoleMLP = CartPoleMLP()

    algorithm: DQN = DQN(environment, network_architecture, DEVICE)

    algorithm.train(
        optimizer=torch.optim.SGD(network_architecture.parameters(), lr=LEARNING_RATE),
        episodes=EPISODES,
        replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=lambda episode: 1 / (1 + episode / 100),
        batch_size=BATCH_SIZE,
    )

    algorithm.run(episodes=2000)