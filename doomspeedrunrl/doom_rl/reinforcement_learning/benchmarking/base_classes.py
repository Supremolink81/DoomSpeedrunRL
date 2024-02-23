import abc
from typing import Any

class RealTimeGraph(abc.ABC):

    """
    An abstract class representing real time graphs.

    Has an update rule for continuous updates as new data from 

    an RL pipeline comes in.

    Fields:

        `dict[str, Any]` graph_settings: the settings for the graph.
    """

    graph_settings: dict[str, Any]

    @abc.abstractmethod
    def __init__(self, graph_settings: dict[str, Any]):

        self.graph_settings = graph_settings

    @abc.abstractmethod
    def update(self, info: dict[str, Any]) -> None:

        """
        Updates the graph using info from the RL pipeline.

        Arguments:

            `dict[str, Any]` info: the info to use to update the graph.
        """

    @abc.abstractmethod
    def graph(self) -> None:

        """
        Displays the contents of the graph in a window.
        """