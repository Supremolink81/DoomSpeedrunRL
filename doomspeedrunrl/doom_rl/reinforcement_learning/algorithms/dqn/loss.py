import torch
from doom_rl.reinforcement_learning.algorithms.base_classes import Transition

def loss_function(q_function: torch.nn.Module, batch: list[Transition], discount_factor: float, device: torch.device) -> torch.Tensor:

    """
    Loss function for Deep Q Learning, which is simply the

    Mean Squared Error between the projected return of the current

    state and the projected return yielded by the reward and 
    
    discounted return of the next state.

    Arguments:

        `torch.nn.Module` q_function: the Q function to use for return estimation.

        `list[Transition]` batch: the batch of state transitions to use to calculate the loss.

        `float` discount_factor: the discount factor for the projected return.

        `torch.device` device: the device to use for tensors.
    
    Returns:

        a PyTorch tensor with the loss value.
    """

    current_states, rewards, actions, next_states, terminal_mask = zip(*batch)

    current_states_tensor: torch.Tensor = torch.stack(current_states, dim=0).to(device)
    
    # int64 needed for indexing tensors
    actions_tensor: torch.Tensor = torch.tensor(actions).type(torch.int64).to(device)

    rewards_tensor: torch.Tensor = torch.tensor(rewards).to(device)
    
    next_states_tensor: torch.Tensor = torch.stack(next_states, dim=0).to(device)

    terminal_mask_tensor: torch.Tensor = torch.tensor(terminal_mask).to(device)

    # this trick ensures we can index the q values using the actions taken
    q_values: torch.Tensor = q_function(current_states_tensor)

    # reshaping to be compatible with torch.gather
    actions_tensor = actions_tensor.reshape((-1, 1))

    q_values_for_actions: torch.Tensor = torch.gather(q_values, dim=1, index=actions_tensor).reshape((-1))

    a = q_function(next_states_tensor)

    optimal_next_state_values: torch.Tensor = torch.max(a, axis=-1)[0]

    target_q_values: torch.Tensor = rewards_tensor + discount_factor * (optimal_next_state_values * terminal_mask_tensor)

    return torch.nn.functional.mse_loss(q_values_for_actions, target_q_values)