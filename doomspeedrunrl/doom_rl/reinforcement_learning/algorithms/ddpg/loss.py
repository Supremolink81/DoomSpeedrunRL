import torch
from doom_rl.reinforcement_learning.algorithms.base_classes import Transition

def loss_function_ddpg_critic(
    critic_network: torch.nn.Module, 
    target_critic_network: torch.nn.Module,
    target_actor_network: torch.nn.Module, 
    batch: list[Transition], 
    discount_factor: float, 
    device: torch.device
) -> torch.Tensor:

    """
    Loss function for the DDPG algorithm's critic, which is simply

    the mean squared error between the projected return of 

    the current state and the projected return from
     
    the reward plus the discounted return of the next state. 

    For calculation of the current state's return, the critic

    network is used, whereas for the next state's return, the target

    actor and critic networks are used. 

    Arguments:

        `torch.nn.Module` critic_network: the critic network to use for the loss.

        `torch.nn.Module` target_critic_network: the target critic network to use for the loss.

        `torch.nn.Module` target_critic_network: the target actor network to use for the loss.

        `list[Transition]` batch: the batch of state transitions to use to calculate the loss.

        `float` discount_factor: the discount factor for the projected return.

        `torch.device` device: the device to use for tensors.

    Returns:

        a PyTorch tensor with the loss value.
    """

    current_states, rewards, actions, next_states, terminal_mask = zip(*batch)

    current_states_tensor: torch.Tensor = torch.stack(current_states, dim=0).to(device)
    
    # int64 needed for indexing tensors
    actions_tensor: torch.Tensor = torch.stack(actions, dim=0).to(device)

    rewards_tensor: torch.Tensor = torch.tensor(rewards).to(device)
    
    next_states_tensor: torch.Tensor = torch.stack(next_states, dim=0).to(device)

    states_actions_tensor: torch.Tensor = torch.cat((current_states_tensor, actions_tensor), axis=1)

    terminal_mask_tensor: torch.Tensor = torch.tensor(terminal_mask).to(device)

    # reshaping to remove the (, 1) at the end and prevent broadcasting issues
    critic_q_values: torch.Tensor = critic_network(states_actions_tensor).reshape((-1))

    target_actor_actions: torch.Tensor = target_actor_network(next_states_tensor)

    target_states_actions_tensor: torch.Tensor = torch.cat((next_states_tensor, target_actor_actions), axis=1)

    # reshaping to remove the (, 1) at the end and prevent broadcasting issues
    target_critic_q_values: torch.Tensor = target_critic_network(target_states_actions_tensor).reshape((-1))

    discounted_return: torch.Tensor = rewards_tensor + discount_factor * (target_critic_q_values * terminal_mask_tensor)

    return torch.nn.functional.mse_loss(critic_q_values, discounted_return)

def loss_function_ddpg_actor(
    critic_network: torch.nn.Module,
    batch: list[Transition], 
    device: torch.device,  
) -> torch.Tensor:
    
    """
    Loss function for the DDPG algorithm's actor, which is simply

    the average projected return over a collection of state-action pairs.

    Arguments:

        `torch.nn.Module` critic_network: the critic network to use for the loss.

        `list[Transition]` batch: the batch of state transitions to use to calculate the loss.

        `torch.device` device: the device to use for tensors.

    Returns:

        a PyTorch tensor with the loss value.
    """

    current_states, _, actions, _, _ = zip(*batch)

    current_states_tensor: torch.Tensor = torch.stack(current_states, dim=0).to(device)
    
    # int64 needed for indexing tensors
    actions_tensor: torch.Tensor = torch.stack(actions, dim=0).to(device)

    states_actions_tensor: torch.Tensor = torch.cat((current_states_tensor, actions_tensor), axis=1)

    critic_q_values: torch.Tensor = critic_network(states_actions_tensor)

    return -torch.mean(critic_q_values)