import torch

def state_value_function(state_action_value_function: torch.nn.Module, softmax_policy: torch.nn.Module, state: torch.Tensor) -> torch.Tensor:
    
    """
    Uses the state-action value function and a discrete softmax policy

    to calculate the state value function.

    Arguments:

        `torch.nn.Module` state_action_value_function: the state-action value function.

        `torch.nn.Module` softmax_policy: the softmax policy.

        `torch.Tensor` state: the environment state.

    Returns:

        a torch.Tensor with a value representing the state value function's output.
    """

    action_distribution: torch.Tensor = softmax_policy(state)

    action_values: torch.Tensor = state_action_value_function(state)

    return torch.sum(action_values * action_distribution)