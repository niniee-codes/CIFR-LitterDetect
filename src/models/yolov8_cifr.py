import json
from pathlib import Path
import torch
import torch.nn as nn
from ultralytics import YOLO

try:
    from src.models.cifr import CIFRBlock, load_class_weights
except ModuleNotFoundError:
    from cifr import CIFRBlock, load_class_weights



class YOLOv8WithCIFR(nn.Module):
    """
    YOLOv8 Model Integrated with Class-Imbalance-Aware Feature Recalibration (CIFR).

    Inserts CIFR blocks via a PyTorch forward pre-hook on the Detect head (model.model[-1]).
    Intercepts the multi-scale feature maps (P3, P4, P5) before they enter the Detect head,
    applies scale-specific CIFR recalibration, and passes the modified feature maps into
    the Detect head seamlessly without requiring structural network surgery.

    Args:
        model_variant (str): YOLOv8 variant weights path or name (e.g. 'yolov8n.pt', 'yolov8m.pt').
        class_weights_json_path (str or Path): Path to class_weights.json file.
        weighting_scheme (str): Weighting scheme to load ('inv_sqrt_freq', 'inv_freq', 'inv_log_freq').
        reduction_ratio (int): Channel reduction ratio for squeeze step in CIFR blocks (default: 16).
        verbose_hook (bool): Whether the pre-hook prints logging info when executing (default: True).
    """

    def __init__(
        self,
        model_variant: str = "yolov8n.pt",
        class_weights_json_path: str = "results/tables/class_weights.json",
        weighting_scheme: str = "inv_sqrt_freq",
        reduction_ratio: int = 16,
        verbose_hook: bool = True,
    ):
        super().__init__()
        self.model_variant = model_variant
        self.class_weights_json_path = class_weights_json_path
        self.weighting_scheme = weighting_scheme
        self.reduction_ratio = reduction_ratio
        self.verbose_hook = verbose_hook

        # 1. Load ultralytics YOLO model
        self.yolo = YOLO(model_variant)
        # Expose underlying PyTorch model as self.model
        self.model = self.yolo.model

        # 2. Load class weights
        self.class_weights = load_class_weights(
            class_weights_json_path, weighting_scheme=weighting_scheme
        )

        # 3. Identify Detect head and its multi-scale channel sizes (P3, P4, P5)
        self.detect_head = self.model.model[-1]
        if hasattr(self.detect_head, "ch"):
            scale_channels = list(self.detect_head.ch)
        else:
            scale_channels = [
                cv2[0].conv.in_channels for cv2 in self.detect_head.cv2
            ]

        # 4. Create one CIFRBlock per scale (3 total)
        self.cifr_blocks = nn.ModuleList(
            [
                CIFRBlock(
                    num_channels=ch,
                    class_weights=self.class_weights,
                    reduction_ratio=reduction_ratio,
                )
                for ch in scale_channels
            ]
        )

        # 5. Register forward pre-hook on Detect head
        self._hook_handle = self.detect_head.register_forward_pre_hook(
            self._cifr_pre_hook
        )

    def _cifr_pre_hook(self, module, args):
        if not args:
            return args

        x = args[0]
        if not isinstance(x, (tuple, list)):
            return args

        recalibrated = []
        for i, feature_map in enumerate(x):
            if i < len(self.cifr_blocks):
                out_map = self.cifr_blocks[i](feature_map)
                if self.verbose_hook:
                    print(
                        f"[CIFR Pre-Hook] Scale P{i+3} (channels={self.cifr_blocks[i].num_channels}): "
                        f"feature map {feature_map.shape} recalibrated -> {out_map.shape}"
                    )
                recalibrated.append(out_map)
            else:
                recalibrated.append(feature_map)

        if isinstance(x, tuple):
            recalibrated = tuple(recalibrated)

        return (recalibrated,) + args[1:]

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):

        """
        Sets PyTorch training/evaluation mode across all submodules
        without triggering Ultralytics' dataset training pipeline.
        """
        for module in self.modules():
            object.__setattr__(module, "training", mode)
        return self


    def train_yolo(self, *args, **kwargs):
        """Launches Ultralytics YOLO training pipeline explicitly."""
        return self.yolo.train(*args, **kwargs)

    def val(self, *args, **kwargs):
        return self.yolo.val(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.yolo.predict(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.yolo, name)



if __name__ == "__main__":
    print("--- Testing YOLOv8WithCIFR Integration ---")
    weights_path = Path("results/tables/class_weights.json")

    # Instantiate wrapper with yolov8n.pt
    cifr_yolo = YOLOv8WithCIFR(
        model_variant="yolov8n.pt",
        class_weights_json_path=weights_path,
        weighting_scheme="inv_sqrt_freq",
        reduction_ratio=16,
        verbose_hook=True,
    )

    # Single dummy forward pass
    dummy_img = torch.randn(1, 3, 640, 640)
    print(f"Input image tensor shape: {dummy_img.shape}")

    cifr_yolo.model.eval()
    with torch.no_grad():
        output = cifr_yolo.model(dummy_img)


    print("\n--- Model Output Summary ---")
    if isinstance(output, (tuple, list)):
        for i, out_item in enumerate(output):
            if isinstance(out_item, torch.Tensor):
                print(f"Output element {i} shape: {out_item.shape}")
            elif isinstance(out_item, (tuple, list)):
                shapes = [t.shape if isinstance(t, torch.Tensor) else type(t) for t in out_item]
                print(f"Output element {i} nested shapes: {shapes}")
    elif isinstance(output, torch.Tensor):
        print(f"Final output tensor shape: {output.shape}")
    else:
        print(f"Final output type: {type(output)}")

    print("\nYOLOv8 + CIFR integration test passed successfully!")
