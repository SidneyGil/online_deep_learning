"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return torch.nn.functional.cross_entropy(logits, target)  # Compute loss


class LinearClassifier(nn.Module):
    def __init__(self, h: int = 64, w: int = 64, num_classes: int = 6):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        layers = []
        layers.append(nn.Flatten())  # Flatten image
        layers.append(nn.Linear(h * w * 3, num_classes))  # Fully connected layer
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)  # Forward pass


class MLPClassifier(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.LayerNorm(out_channels),
                torch.nn.ReLU(),
            )  # Apply normalization and activation

            if in_channels != out_channels:
                self.skip = torch.nn.Linear(in_channels, out_channels)  # Adjust dimensions if needed
            else:
                self.skip = torch.nn.Identity()  # No transformation

        def forward(self, x):
            return self.model(x) + self.skip(x)  # Add residual connection

    def __init__(self, h: int = 64, w: int = 64, num_classes: int = 6, hidden_dim: int = 128):
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())  # Flatten input
        layers.append(torch.nn.Linear(h * w * 3, hidden_dim))  # Initial dense layer
        layers.append(self.Block(hidden_dim, hidden_dim))  # Apply residual block
        layers.append(torch.nn.Linear(hidden_dim, num_classes))  # Output layer
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Forward pass


class MLPClassifierDeep(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.LayerNorm(out_channels),
                torch.nn.ReLU(),
            )  # Basic fully connected block

        def forward(self, x):
            return self.model(x)  # Pass input through the block

    def __init__(self, h: int = 64, w: int = 64, num_classes: int = 6, num_layers: int = 4, hidden_dim: int = 128):
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())  # Flatten input
        layers.extend([torch.nn.Linear(h * w * 3, hidden_dim), torch.nn.ReLU()])  # First layer
        layers.extend([self.Block(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])  # Stack multiple blocks
        layers.append(torch.nn.Linear(hidden_dim, num_classes))  # Final output layer
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Forward pass


class MLPClassifierDeepResidual(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.LayerNorm(out_channels),
                torch.nn.ReLU(),
            )  # Fully connected block with activation

            if in_channels != out_channels:
                self.skip = torch.nn.Linear(in_channels, out_channels)  # Adjust residual connection
            else:
                self.skip = torch.nn.Identity()  # Identity mapping

        def forward(self, x):
            return self.model(x) + self.skip(x)  # Add skip connection

    def __init__(self, h: int = 64, w: int = 64, num_classes: int = 6, num_layers: int = 6, hidden_dim: int = 128):
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())  # Flatten input
        in_channels = h * w * 3
        layers.append(nn.Linear(in_channels, hidden_dim))  # Initial layer
        layers.extend([self.Block(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])  # Apply residual blocks
        layers.append(torch.nn.Linear(hidden_dim, num_classes))  # Output layer
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Forward pass


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # Convert parameters to MB


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")  # Save model state
    raise ValueError(f"Model type '{str(type(model))}' not supported")  # Handle unsupported models


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"  # Ensure model file exists
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))  # Load model weights
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")  # Enforce size limit
    print(f"Model size: {model_size_mb:.2f} MB")  # Log model size

    return r  # Return the model
