import plotly
from typing import Any
from doom_rl.reinforcement_learning.benchmarking.base_classes import RealTimeGraph

class MovingAverageGraph(RealTimeGraph):

    """
    A line graph representing a reward average over time.

    Fields:

        `int` history_length: the number of samples to collect from

        the front when averaging. For example, if history_length = 20,

        then the graph uses the 20 most recent samples to calculate the average.

        `list[float]` rewards: the list of rewards.

        `list[float]` averages: the list of averages.

        `plotly.graph_objs.Figure` figure: the figure where the

        moving average is plotted.
    """

    history_length: int

    rewards: list[float]

    averages: list[float]

    figure: plotly.graph_objs.Figure

    def __init__(self, history_length: int, graph_settings: dict[str, Any]):

        self.history_length = history_length

        super().__init__(graph_settings)

        self.rewards = []

        self.averages = []

        self.figure = plotly.graph_objs.Figure()

        empty_scatter: plotly.graph_objs.Scatter = plotly.graph_objs.Scatter(x=[], y=[])

        self.figure.add_trace(empty_scatter)

    def update(self, info: dict[str, Any]) -> None:

        reward: float = info["reward"]

        self.rewards.append(reward)

        new_average: float

        if len(self.averages) == 0:

            self.averages.append(reward)

        elif len(self.averages) < self.history_length:

            previous_average: float = self.averages[-1]

            new_average = (len(self.averages)*previous_average + reward) / (len(self.averages) + 1)
            
        else:

            previous_average: float = self.averages[-1]

            value_to_be_removed: float = self.rewards[-self.history_length]

            new_average = previous_average + (reward - value_to_be_removed) / (self.history_length)
        
        self.averages.append(new_average)

        line_chart: plotly.graph_objs.Scatter = self.figure.data[0]

        line_chart.y += [reward]

    def graph(self) -> None:

        self.figure.show()