import torch
from sklearn.preprocessing import MinMaxScaler


def preprocess_trajectories(trajectories):
    """
    Preprocess trajectories by normalizing and converting them to PyTorch tensors.

    Args:
        trajectories (numpy.ndarray): Raw trajectory data of shape [episodes, timesteps, features].

    Returns:
        torch.Tensor: Normalized trajectories of shape [episodes, timesteps, features].
    """
    # Flatten episodes and timesteps for normalization
    episodes, timesteps, features = trajectories.shape
    flattened = trajectories.reshape(-1, features)

    # Normalize features to [0, 1]
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(flattened)

    # Back to original dimensions
    normalized_trajectories = normalized.reshape(episodes, timesteps, features)

    # Convert to Tensor
    tensor_trajectories = torch.tensor(normalized_trajectories, dtype=torch.float32)

    return tensor_trajectories