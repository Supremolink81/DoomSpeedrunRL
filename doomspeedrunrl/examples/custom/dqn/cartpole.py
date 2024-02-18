import torch
import gymnasium
from doom_rl.reinforcement_learning.algorithms.dqn.agents import DQN
from doom_rl.reinforcement_learning.architectures.basic import CartPoleMLP

LEARNING_RATE = 0.001

EPISODES = 100000

REPLAY_BUFFER_CAPACITY = 500

DISCOUNT_FACTOR = 0.99

EPSILON = 0.1

BATCH_SIZE = 50

if __name__ == "__main__":

    environment: gymnasium.Env = gymnasium.make("CartPole-v1", render_mode="human")

    network_architecture: CartPoleMLP = CartPoleMLP()

    algorithm: DQN = DQN(environment, network_architecture)

    algorithm.train(
        optimizer=torch.optim.SGD(network_architecture.parameters(), lr=LEARNING_RATE),
        episodes=EPISODES,
        replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        batch_size=BATCH_SIZE,
    )

    # algorithm.run(episodes=20)