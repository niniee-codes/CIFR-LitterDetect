import json
from pathlib import Path
import torch
import torch.nn as nn


class CIFRBlock(nn.Module):
    """
    Class-Imbalance-Aware Feature Recalibration (CIFR) Block.

    Adapts Squeeze-and-Excitation (SE) attention by incorporating a class-frequency
    prior bias before the sigmoid activation. This allows the network to recalibrate
    feature maps dynamically based on spatial context squeezed from input feature maps
    and dataset class imbalance frequency priors.

    Args:
        num_channels (int): Number of input feature map channels.
        class_weights (list or torch.Tensor): Precomputed class weights (one per class).
        reduction_ratio (int): Channel reduction ratio for squeeze-excitation (default: 16).
    """

    def __init__(self, num_channels: int, class_weights, reduction_ratio: int = 16):
        super().__init__()
        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio

        if isinstance(class_weights, (list, tuple)):
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        elif isinstance(class_weights, torch.Tensor):
            weights_tensor = class_weights.float()
        else:
            raise TypeError("class_weights must be a list, tuple, or torch.Tensor")

        self.register_buffer("class_weights", weights_tensor)
        num_classes = weights_tensor.numel()

        # Project class_weights to channel dimension if shapes differ
        if num_classes == num_channels:
            self.prior_proj = None
        else:
            self.prior_proj = nn.Linear(num_classes, num_channels, bias=False)

        reduced_channels = max(1, num_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_channels, reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, num_channels)

        # Learnable scalar scaling frequency prior influence
        self.prior_strength = nn.Parameter(torch.tensor(1.0))

    _warned_attn_clamp = False
    _warned_prior_strength_clamp = False

    def get_prior_bias(self) -> torch.Tensor:
        if self.prior_proj is not None:
            return self.prior_proj(self.class_weights)
        return self.class_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # 1. Global Average Pooling (Squeeze)
        squeeze = self.avg_pool(x).view(b, c)

        # 2. Excitation FC layers
        raw_attn = self.fc2(self.relu(self.fc1(squeeze)))

        # 3. Frequency-prior bias addition scaled by learnable prior_strength
        prior_bias = self.get_prior_bias().unsqueeze(0)  # Shape (1, C)

        # Safeguard 2: Constrain prior_strength to a safe range [0.1, 3.0]
        if not CIFRBlock._warned_prior_strength_clamp and (
            self.prior_strength.item() < 0.1 or self.prior_strength.item() > 3.0
        ):
            print(
                f"[WARNING] [CIFRBlock] prior_strength ({self.prior_strength.item():.4f}) "
                f"was outside safe range [0.1, 3.0] and was clamped."
            )
            CIFRBlock._warned_prior_strength_clamp = True

        clamped_prior_strength = torch.clamp(self.prior_strength, min=0.1, max=3.0)
        raw_attn = raw_attn + clamped_prior_strength * prior_bias

        # Safeguard 1: Clamp raw attention scores to safe range [-10.0, 10.0] before sigmoid
        if not CIFRBlock._warned_attn_clamp and (
            (raw_attn < -10.0).any() or (raw_attn > 10.0).any()
        ):
            print(
                f"[WARNING] [CIFRBlock] Raw attention scores were outside safe range [-10.0, 10.0] "
                f"(min={raw_attn.min().item():.4f}, max={raw_attn.max().item():.4f}) and were clamped."
            )
            CIFRBlock._warned_attn_clamp = True

        raw_attn = torch.clamp(raw_attn, min=-10.0, max=10.0)

        # 4. Sigmoid excitation & Recalibration
        attn_weights = torch.sigmoid(raw_attn).view(b, c, 1, 1)
        return x * attn_weights


def load_class_weights(json_path, weighting_scheme: str = "inv_sqrt_freq", class_names=None):
    """
    Reads class_weights.json and returns a list of float weights in class-index order.

    Args:
        json_path (str or Path): Path to class_weights.json file.
        weighting_scheme (str): Key for weighting scheme ('inv_freq', 'inv_sqrt_freq', 'inv_log_freq').
        class_names (list, optional): List of class names defining index order. If None, uses key order in JSON.

    Returns:
        list of float: Class weights ordered by class index.
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if class_names is None:
        class_names = list(data.keys())

    weights = [float(data[name][weighting_scheme]) for name in class_names]
    return weights


if __name__ == "__main__":
    print("--- Testing CIFRBlock ---")
    weights_path = Path("results/tables/class_weights.json")

    if weights_path.exists():
        weights = load_class_weights(weights_path, weighting_scheme="inv_sqrt_freq")
        print(f"Loaded class weights from {weights_path}: {weights}")
    else:
        weights = [0.029, 0.082, 0.057, 0.075, 0.058, 0.085, 0.176, 0.073]
        print(f"Using fallback dummy class weights: {weights}")

    num_channels = 256
    cifr = CIFRBlock(num_channels=num_channels, class_weights=weights, reduction_ratio=16)

    # 1. Create dummy input tensor
    dummy_input = torch.randn(2, num_channels, 20, 20)
    print(f"Input shape: {dummy_input.shape}")

    # 2. Forward pass
    output = cifr(dummy_input)
    print(f"Output shape: {output.shape}")
    assert dummy_input.shape == output.shape, "Input and output shapes do not match!"
    print("Shapes match successfully!")

    # 3. Forward + Backward test
    loss = output.sum()
    loss.backward()

    print(f"Gradient for prior_strength: {cifr.prior_strength.grad}")
    assert cifr.prior_strength.grad is not None, "Gradient did not flow to prior_strength!"
    print("Module forward + backward pass completed without error!")
