import torch

def advantage_estimates() -> torch.Tensor:

    """
    Computes the advantage estimates over a set of timesteps using

    a sequence of rewards and a state value function.


    """