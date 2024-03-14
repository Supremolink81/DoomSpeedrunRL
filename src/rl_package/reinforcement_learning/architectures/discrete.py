import torch

class CartPoleMLP(torch.nn.Module):

    """
    A simple MLP architecture for use in the CartPole-v1 gym environment. 
    """

    first_linear_layer: torch.nn.Linear

    second_linear_layer: torch.nn.Linear

    third_linear_layer: torch.nn.Linear

    fourth_linear_layer: torch.nn.Linear

    gelu_activation: torch.nn.GELU

    def __init__(self):

        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU6(),
            torch.nn.Linear(16, 64),
            torch.nn.ReLU6(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU6(),
            torch.nn.Linear(16, 4),
            torch.nn.ReLU6(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, cartpole_state: torch.Tensor) -> torch.Tensor:

        return self.model(cartpole_state)
    
class AcrobotMLP(torch.nn.Module):

    """
    A simple MLP architecture for use in the Acrobot-v1 gym environment.
    """

    model: torch.nn.Sequential

    def __init__(self):

        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(6, 5),
            torch.nn.GELU(),
            torch.nn.Linear(5, 4),
            torch.nn.GELU(),
            torch.nn.Linear(4, 3),
            torch.nn.GELU(),
        )

    def forward(self, acrobot_state: torch.Tensor) -> torch.Tensor:

        return self.model(acrobot_state)