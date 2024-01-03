import abc
import gymnasium
import torch
import numpy as np
import cupy
from collections import deque
import random
from typing import Any, Dict, Union

ArrayType = Union[np.array, torch.Tensor, cupy.array, int]

Transition = tuple[ArrayType, int, float, ArrayType]

class RLPipeline(abc.ABC):

    """
    Base class for an RL pipeline. Is kept as open

    as possible to accomodate the wide variety of possible

    RL algorithms (e.g. DQN, DDPG, PPO, A2C).

    Fields:

        `gymnasium.Env` environment: the enviroment the agent resides in.
    """

    environment: gymnasium.Env
    
    @abc.abstractmethod
    def __init__(self, environment: gymnasium.Env):

        self.environment = environment

    @abc.abstractmethod
    def train(self, **kwargs: Dict[str, Any]) -> None:
        """
        Trains the agent using the given environment. 

        Arguments:

            `gymnasium.Env` environment: the environment to train in.

            `Dict[str, Any]` kwargs: keyword arguments for specific
            agent types (examples include: the discount factor, the
            learning rate, and the replay buffer size).
        """

    @abc.abstractmethod
    def run(self, episodes: int = -1) -> None:
        """
        Run a given number of episodes in the environment with a policy.

        Arguments:

            `int` episodes: the number of episodes to run. Defaults
            to -1, i.e. runs episodes until the user quits.
        """

class ReplayBuffer:

    """
    Class used for experience replay in RL.

    Fields:

        `deque` storage: where to store previous experiences.
    """

    storage: deque

    def __init__(self, capacity: int):

        """
        Initializes the replay buffer with a given capacity.

        Arguments:

            `int` capacity: the capacity of the replay buffer.
        """

        self.storage = deque([], maxlen=capacity)

    def add(self, transition: Transition) -> None:

        """
        Adds a state transition into the replay buffer.

        Arguments:

            `Transition` transition: the transition to add.
        """

        self.storage.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:

        return random.sample(self.storage, batch_size)