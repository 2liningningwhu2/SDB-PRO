import torch
import torch.nn as nn


class SDBCNN(nn.Module):
    """
    CNN baseline for Satellite-Derived Bathymetry.

    Input
    -----
    x : [B, 4, 5, 5]

    Output
    ------
    output : [B, 2]

    output[:, 0] : predicted depth
    output[:, 1] : log variance
    """

    def __init__(
        self,
        in_channels=4,
        patch_size=5
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            )
        )

        # --------------------------------------------------
        # 自动计算 CNN 输出尺寸
        # --------------------------------------------------

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                in_channels,
                patch_size,
                patch_size
            )

            feature_output = self.features(
                dummy
            )

            flatten_dim = feature_output.view(
                1, -1
            ).shape[1]

        self.regressor = nn.Sequential(
            nn.Linear(
                flatten_dim,
                16
            ),
            nn.ReLU(),

            nn.Linear(
                16,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        output = self.regressor(x)

        return output


if __name__ == "__main__":

    model = SDBCNN(
        in_channels=4,
        patch_size=5
    )

    print("========== Model Test ==========")

    print(model)

    # 模拟一个 DataLoader batch
    x = torch.randn(
        128,
        4,
        5,
        5
    )

    output = model(x)

    print(
        f"\nInput shape: "
        f"{x.shape}"
    )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print(
        f"Depth output shape: "
        f"{output[:, 0].shape}"
    )

    print(
        f"Log variance shape: "
        f"{output[:, 1].shape}"
    )

    print(
        f"\nNumber of parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )