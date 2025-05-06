from torch.utils.data import DataLoader, TensorDataset
import torch


def prepare_trajectories_for_ddpm(trajectories, batch_size):
    """
    Prepare trajectories for DDPM training by flattening and batching.

    Args:
        trajectories (torch.Tensor): Trajectories of shape [episodes, timesteps, features].
        batch_size (int): The size of each batch for training.

    Returns:
        DataLoader: A PyTorch DataLoader with batches of flattened trajectories.
    """
    # Flatten [episodes, timesteps] into a single dimension
    episodes, timesteps, features = trajectories.shape
    flattened = trajectories.reshape(-1, features)  # Shape: [episodes * timesteps, features]

    dataset = TensorDataset(flattened)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader