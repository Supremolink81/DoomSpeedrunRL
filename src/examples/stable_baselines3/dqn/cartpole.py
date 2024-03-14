import stable_baselines3
import gymnasium
import numpy as np

if __name__ == "__main__":

    environment: gymnasium.Env = gymnasium.make("BipedalWalker-v3", render_mode="human", max_episode_steps=5000)

    algorithm: stable_baselines3.PPO = stable_baselines3.A2C("MlpPolicy", env=environment, learning_rate=0.0001)

    DESIRED_EPISODES = 10000000000

    TIMESTEPS_PER_EPISODE = 5000

    algorithm.learn(total_timesteps=TIMESTEPS_PER_EPISODE * DESIRED_EPISODES)

    state: np.array = algorithm.get_env().reset()

    while True:

        action, _ = algorithm.predict(state)

        next_state, _, _, _ = algorithm.get_env().step(action)

        state = next_state

        algorithm.get_env().render("human")