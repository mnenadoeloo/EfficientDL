import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Sequential CNN (no skip connections), FP32, eval-only.

    Conv -> ReLU blocks, padding=k//2, bias=False on every conv.
    No BatchNorm: matches the layer table in the assignment exactly, and
    keeps this model consistent with the FLOPs/Memory formulas derived by
    hand in derivation.tex (which do not account for BN params/FLOPs).
    Input: 3 x S x S. Output: 100 classes.
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()

        def conv_relu(c_in, c_out, k, stride):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride,
                          padding=k // 2, bias=False),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_relu(3, 32, 7, 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_relu(32, 64, 5, 1),
            conv_relu(64, 128, 3, 2),
            conv_relu(128, 256, 1, 1),
            conv_relu(256, 256, 3, 2),
            conv_relu(256, 512, 1, 1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


def build_model(num_classes: int = 100) -> nn.Module:
    return SmallCNN(num_classes=num_classes)


if __name__ == "__main__":
    m = build_model().eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        y = m(x)
    print(y.shape)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params}")
