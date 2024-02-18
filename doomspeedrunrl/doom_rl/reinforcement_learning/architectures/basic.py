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

        self.first_linear_layer = torch.nn.Linear(4, 4)

        self.second_linear_layer = torch.nn.Linear(4, 3)

        self.third_linear_layer = torch.nn.Linear(3, 3)

        self.fourth_linear_layer = torch.nn.Linear(3, 2)

        self.gelu_activation = torch.nn.GELU()

    def forward(self, cartpole_state: torch.Tensor) -> torch.Tensor:

        output_tensor_1: torch.Tensor = self.first_linear_layer(cartpole_state)

        output_tensor_2: torch.Tensor = self.gelu_activation(output_tensor_1)

        output_tensor_3: torch.Tensor = self.second_linear_layer(output_tensor_2 + cartpole_state)

        output_tensor_4: torch.Tensor = self.gelu_activation(output_tensor_3)

        output_tensor_5: torch.Tensor = self.third_linear_layer(output_tensor_4 + output_tensor_3)

        output_tensor_6: torch.Tensor = self.gelu_activation(output_tensor_5)

        output_tensor_7: torch.Tensor = self.fourth_linear_layer(output_tensor_6)

        return self.gelu_activation(output_tensor_7)
    
class AcrobotMLP(torch.nn.Module):

    """
    A simple MLP architecture for use in the Acrobot-v1 gym environment.
    """

    first_linear_layer: torch.nn.Linear

    second_linear_layer: torch.nn.Linear

    third_linear_layer: torch.nn.Linear

    gelu_activation: torch.nn.GELU

    def __init__(self):

        super().__init__()

        self.first_linear_layer = torch.nn.Linear(6, 5)

        self.second_linear_layer = torch.nn.Linear(5, 4)

        self.third_linear_layer = torch.nn.Linear(4, 3)

        self.gelu_activation = torch.nn.GELU()

    def forward(self, acrobot_state: torch.Tensor) -> torch.Tensor:

        output_tensor_1: torch.Tensor = self.first_linear_layer(acrobot_state)

        output_tensor_2: torch.Tensor = self.gelu_activation(output_tensor_1)

        output_tensor_3: torch.Tensor = self.second_linear_layer(output_tensor_2)

        output_tensor_4: torch.Tensor = self.gelu_activation(output_tensor_3)

        output_tensor_5: torch.Tensor = self.third_linear_layer(output_tensor_4)

        return self.gelu_activation(output_tensor_5)