import torch
import gymnasium
from rl_package.reinforcement_learning.algorithms.dqn.agents import DQN
from rl_package.reinforcement_learning.architectures.discrete import AcrobotMLP

LEARNING_RATE = 0.001

EPISODES = 10

REPLAY_BUFFER_CAPACITY = 500

DISCOUNT_FACTOR = 0.99

BATCH_SIZE = 50

DEVICE = torch.device("cuda:0")

if __name__ == "__main__":

    environment: gymnasium.Env = gymnasium.make("Acrobot-v1")

    network_architecture: AcrobotMLP = AcrobotMLP()

    algorithm: DQN = DQN(environment, network_architecture, DEVICE)

    algorithm.train(
        optimizer=torch.optim.SGD(network_architecture.parameters(), lr=LEARNING_RATE),
        episodes=EPISODES,
        replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=lambda episode: 1 / (1 + episode),
        batch_size=BATCH_SIZE,
    )

    algorithm.run(episodes=2000)