import abc
import gymnasium
import torch
import numpy as np
from collections import deque
import random
from typing import Any, Dict, Union, Iterable, Optional, Callable
from rl_package.reinforcement_learning.benchmarking.base_classes import MetricTracker

ArrayType = Union[np.array, torch.Tensor]

Transition = tuple[ArrayType, int, float, ArrayType]

ActionType = Union[ArrayType, int]

class RLPipeline(abc.ABC):

    """
    Base class for an RL pipeline. Is kept as open

    as possible to accomodate the wide variety of possible

    RL algorithms (e.g. DQN, DDPG, PPO, A2C).

    Fields:

        `torch.device` device: the device to use for training and inference. 
    """

    device: torch.device

    @abc.abstractmethod
    def action(self, state: ArrayType, **kwargs: dict[str, Any]) -> ActionType:

        """
        Abstract function to select an action

        Arguments:

            `ArrayType` state: the state to use for the agent.

            `dict[str, Any]` **kwargs: keyword arguments for the policy.

        Returns:

            The action chosen. 
        """

    @abc.abstractmethod
    def train(self, 
        discount_factor: float = 0.99, 
        state_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, 
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Trains the agent using the given environment. 

        Arguments:

            `float` discount_factor: the discount factor for returns; default is 0.99.

            `Optional[Callable[[torch.Tensor], torch.Tensor]]` state_transform: an optional

            function used to transform the state. Defaults to None.

            `Dict[str, Any]` **kwargs: keyword arguments for specific
            agent types (examples include: the discount factor, the
            learning rate, the number of timesteps and the replay buffer size).
        """

    @abc.abstractmethod
    def run(self, episodes: int = -1, **kwargs: Dict[str, Any]) -> None:
        """
        Run a given number of episodes in the environment with a policy.

        Arguments:

            `int` episodes: the number of episodes to run. Defaults
            to -1, i.e. runs episodes until the user quits.

            `Dict[str, Any]` kwargs: keyword arguments for the 
            state transforms and/or other utilities.
        """

    @abc.abstractmethod
    def human_rendering(self) -> None:
        """
        Sets the render mode to human: i.e. makes it where

        the environment is rendered on the screen
        """
    
    @abc.abstractmethod
    def no_rendering(self) -> None:
        """
        Disables rendering of any kind.
        """

    @abc.abstractmethod
    def _update_metrics(self, metrics: Iterable[MetricTracker], info: Dict[str, Any]) -> None:

        """
        Updates a collection of metric trackers using state collected from

        training and/or inference.

        Arguments:

            `Iterable[MetricTracker]` metrics: the metrics to update.

            `Dict[str, Any]` info: the info to use for updating.
        """
        
        for metric in metrics:

            metric.update(info)

class SingleAgentRLPipeline(RLPipeline):

    """
    Base class for a single agent RL pipeline.

    Fields:

        `gymnasium.Env` environment: the environment to train in.
    """

    environment: gymnasium.Env

    @abc.abstractmethod
    def __init__(self, environment: gymnasium.Env):

        self.environment = environment

    def human_rendering(self) -> None:
        
        self.environment = gymnasium.make(self.environment.spec, render_mode="human")

    def no_rendering(self) -> None:
        
        self.environment = gymnasium.make(self.environment.spec, render_mode=None)

class MultiAgentRLPipeline(RLPipeline):

    """
    Base class for a multi agent RL pipeline. 

    Fields:

        `gymnasium.vector.VectorEnv` environment: the environment to train in.
    """

    environment: gymnasium.vector.VectorEnv

    @abc.abstractmethod
    def __init__(self, environment: gymnasium.vector.VectorEnv):

        self.environment = environment

    def epsilon_greedy_action(self, state: ArrayType, epsilon: float) -> int:

        random_number: float = random.random()

        if random_number >= epsilon:

            return self.environment.action_space.sample()

        else:

            action_distribution: torch.Tensor = self.q_function(state.reshape((1,)+state.shape))

            return torch.argmax(action_distribution)[0]

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

        sample_size: int = min(len(self.storage), batch_size)

        return random.sample(self.storage, sample_size)