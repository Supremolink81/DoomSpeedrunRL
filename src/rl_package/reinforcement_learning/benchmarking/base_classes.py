import abc
from typing import Any

class MetricTracker(abc.ABC):

    """
    An abstract class representing metric trackers for RL.

    Has an update rule for continuous updates as new data from 

    an RL pipeline comes in.

    Fields:

        `dict[str, Any]` settings: the settings for the metric tracker.
    """

    settings: dict[str, Any]

    @abc.abstractmethod
    def __init__(self, settings: dict[str, Any]):

        self.settings = settings

    @abc.abstractmethod
    def update(self, info: dict[str, Any]) -> None:

        """
        Updates the metric using info from the RL pipeline.

        Arguments:

            `dict[str, Any]` info: the info to use to update the metric.
        """

class RealTimeGraph(MetricTracker):

    """
    An abstract class representing real time graphs.

    Has an update rule for continuous updates as new data from 

    an RL pipeline comes in.
    """

    graph_settings: dict[str, Any]

    @abc.abstractmethod
    def __init__(self, graph_settings: dict[str, Any]):

        super().__init__(graph_settings)

    @abc.abstractmethod
    def graph(self) -> None:

        """
        Displays the contents of the graph in a window.
        """