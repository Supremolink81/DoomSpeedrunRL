import torch
from torch.utils import data

class AdvantageEstimateDataset(data.Dataset):

    """
    A dataset for storing advantage estimates; for use in the PPO algorithm.

    Fields:

        `torch.Tensor` advantage_estimates: the advantage estimates in the dataset.
    """

    advantage_estimates: torch.Tensor

    def __init__(self, advantage_estimates: torch.Tensor):

        self.advantage_estimates = advantage_estimates

    def __get__(self, index: int) -> float:

        return self.advantage_estimates[index]
    
    def __len__(self) -> int:

        return self.advantage_estimates.shape[0]