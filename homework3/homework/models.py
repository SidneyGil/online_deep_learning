from pathlib import Path

import torch
from torch._prims_common import Tensor
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class RegressionLoss(nn.Module):
  def forward(self, predictions: torch.Tensor, target: torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss() 
    return loss(predictions,target)

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
        loss = torch.nn.CrossEntropyLoss()
        return loss(logits,target)


class Classifier(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU()
            )
            if n_input != n_output:
                self.skip=(torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride, padding=0)) 
            else:
                self.skip = torch.nn.Identity()
               
        def forward(self, x):
            # identity = x
            # if self.downsample is not None:
            #     identity = self.downsample(x)
            # return self.net(x) + identity
            
            return self.skip(x) + self.net(x)
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()
        print("in_channels", in_channels)
        cnn_layers = [
            torch.nn.Conv2d(in_channels,64, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU(),
        ]
        c1=64
        for _ in range(3):
           c2 = c1*2
           cnn_layers.append(self.Block(c1, c2, stride=2))
           c1=c2

          # Adjusted stride for pooling
        
        self.oneconv= (torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        self.globalavgpool=(torch.nn.AdaptiveAvgPool2d(1))

        # cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        # cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))

        self.network = torch.nn.Sequential(*cnn_layers)

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        #print(self.network)

        # TODO: implement


        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        
        x = self.network(x)
        x = self.oneconv(x)

        logits = self.globalavgpool(x)

        return logits.view(logits.size(0), -1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)

class Detector(torch.nn.Module):
    class EncoderBlock(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
        ):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        def forward(self, x):
            return self.conv(x)

    class DecoderBlock(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
        ):
            super().__init__()
            self.conv = nn.Sequential([

                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])

        def forward(self, x):
            return self.conv(x)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        encoder_layers = [
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        ]
        decoder_layers = nn.ModuleList()
        c=32
        for _ in range(3):
            c1=c*2
            encoder_layers.append(self.EncoderBlock(c, c1 ))
            c=c1
        
        c=128
        for _ in range(3):
            c1=c*2
            decoder_layers.append(nn.ConvTranspose2d(c1, c, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.ReLU())
            c=c//2

        self.encoder_network = nn.Sequential(*encoder_layers)
        self.decoder_network = nn.Sequential(*decoder_layers)

        self.seg = nn.Conv2d(32, num_classes, kernel_size=1)
        self.depth = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        encoded_net = self.encoder_network(x)
        decoded_net = self.decoder_network(encoded_net)

        return self.seg(decoded_net), self.depth(decoded_net).squeeze(1)

        
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()