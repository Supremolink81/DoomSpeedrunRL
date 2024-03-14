import torch

class BipedalWalkerActorMLP(torch.nn.Module):

    """
    A simple MLP actor architecture for use in the BipedalWalker-v3 gym environment.
    """

    model: torch.nn.Sequential

    def __init__(self):

        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(24, 21),
            torch.nn.GELU(),
            torch.nn.Linear(21, 18),
            torch.nn.GELU(),
            torch.nn.Linear(18, 15),
            torch.nn.GELU(),
            torch.nn.Linear(15, 12),
            torch.nn.GELU(),
            torch.nn.Linear(12, 9),
            torch.nn.GELU(),
            torch.nn.Linear(9, 6),
            torch.nn.GELU(),
            torch.nn.Linear(6, 4),
            torch.nn.Tanh(),
        )

        # this is so the data type aligns with the environment state's data type
        self.double()

    def forward(self, bipedal_walker_state: torch.Tensor) -> torch.Tensor:

        return self.model(bipedal_walker_state)
    
class BipedalWalkerCriticMLP(torch.nn.Module):

    """
    A simple MLP actor architecture for use in the BipedalWalker-v3 gym environment.
    """

    model: torch.nn.Sequential

    def __init__(self):

        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(28, 25),
            torch.nn.GELU(),
            torch.nn.Linear(25, 22),
            torch.nn.GELU(),
            torch.nn.Linear(22, 19),
            torch.nn.GELU(),
            torch.nn.Linear(19, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 13),
            torch.nn.GELU(),
            torch.nn.Linear(13, 10),
            torch.nn.GELU(),
            torch.nn.Linear(10, 7),
            torch.nn.GELU(),
            torch.nn.Linear(7, 4),
            torch.nn.GELU(),
            torch.nn.Linear(4, 1),
            torch.nn.GELU(),
        )

        # this is so the data type aligns with the environment state's data type
        self.double()

    def forward(self, bipedal_walker_state: torch.Tensor) -> torch.Tensor:

        return self.model(bipedal_walker_state)