import torch
import numba
from doom_rl.reinforcement_learning.algorithms.base_classes import Transition

@numba.jit
def loss_function(q_function: torch.nn.Module, batch: list[Transition], discount_factor: float) -> torch.Tensor:

    """
    Loss function for Deep Q Learning, which is simply the

    Mean Squared Error between the projected return of the current

    state and the projected return yielded by the reward and 
    
    discounted return of the next state.

    Arguments:

        `torch.nn.Module` q_function: the Q function to use for return estimation.

        `list[Transition]` batch: the batch of state transitions to use to calculate the loss.

        `float` discount_factor: the discount factor for the projected return.
    
    Returns:

        a PyTorch tensor with the loss value.
    """

    current_states, actions, rewards, next_states, terminal_mask = zip(*batch)

    current_states_tensor: torch.Tensor = torch.stack(current_states, dim=0)
    
    actions_tensor: torch.Tensor = torch.stack(actions, dim=0)

    rewards_tensor: torch.Tensor = torch.stack(rewards, dim=0)
    
    next_states_tensor: torch.Tensor = torch.stack(next_states, dim=0)

    terminal_mask_tensor: torch.Tensor = torch.stack(terminal_mask, dim=0)

    # this trick ensures we can index the q values using the actions taken
    q_values_for_actions: torch.Tensor = q_function(current_states_tensor)[range(len(current_states)), actions_tensor]

    optimal_next_state_values: torch.Tensor = torch.max(q_function(next_states_tensor), axis=-1)[0]

    target_q_values: torch.Tensor = rewards_tensor + discount_factor * (optimal_next_state_values * terminal_mask_tensor)

    return torch.nn.functional.mse_loss(q_values_for_actions, target_q_values)