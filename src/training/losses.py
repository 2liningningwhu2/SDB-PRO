import torch
import torch.nn as nn


class StableHeteroscedasticLoss(nn.Module):
    """
    Stable heteroscedastic regression loss for SDB.

    Model output:
        prediction[:, 0] -> predicted depth
        prediction[:, 1] -> raw log variance

    The raw log variance is constrained to a stable range
    using tanh before calculating the loss.
    """

    def __init__(
        self,
        log_var_scale=5.0,
        log_var_weight=0.1,
        min_precision=1e-6,
        max_precision=1e6,
    ):
        super().__init__()

        self.log_var_scale = log_var_scale
        self.log_var_weight = log_var_weight
        self.min_precision = min_precision
        self.max_precision = max_precision

    def forward(self, prediction, target):
        """
        Parameters
        ----------
        prediction : torch.Tensor
            Shape [B, 2].
            prediction[:, 0] is depth.
            prediction[:, 1] is raw log variance.

        target : torch.Tensor
            Shape [B].

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """

        if prediction.ndim != 2:
            raise ValueError(
                f"prediction must have shape [B, 2], "
                f"got {prediction.shape}"
            )

        if prediction.shape[1] != 2:
            raise ValueError(
                f"prediction must have 2 outputs, "
                f"got {prediction.shape[1]}"
            )

        if target.ndim != 1:
            raise ValueError(
                f"target must have shape [B], "
                f"got {target.shape}"
            )

        if prediction.shape[0] != target.shape[0]:
            raise ValueError(
                "prediction and target must have "
                "the same batch size"
            )

        # --------------------------------------------------
        # 1. Separate depth prediction and uncertainty
        # --------------------------------------------------
        depth_pred = prediction[:, 0]
        raw_log_var = prediction[:, 1]

        # --------------------------------------------------
        # 2. Constrain log variance to a stable range
        #    [-log_var_scale, +log_var_scale]
        # --------------------------------------------------
        log_var = (
            self.log_var_scale
            * torch.tanh(raw_log_var)
        )

        # --------------------------------------------------
        # 3. Convert log variance to precision
        #
        #    precision = exp(-log variance)
        # --------------------------------------------------
        precision = torch.exp(-log_var)

        precision = torch.clamp(
            precision,
            min=self.min_precision,
            max=self.max_precision,
        )

        # --------------------------------------------------
        # 4. Squared regression error
        # --------------------------------------------------
        squared_error = (
            depth_pred - target
        ) ** 2

        # --------------------------------------------------
        # 5. Heteroscedastic loss
        #
        #    high uncertainty -> lower precision
        #    low uncertainty  -> higher precision
        # --------------------------------------------------
        loss = (
            precision * squared_error
            + self.log_var_weight * log_var
        )

        return loss.mean()


if __name__ == "__main__":

    print("========== Loss Test ==========")

    batch_size = 128

    # 模拟 CNN 输出
    prediction = torch.randn(
        batch_size,
        2,
        requires_grad=True
    )

    # 模拟真实水深
    target = torch.randn(
        batch_size
    )

    criterion = StableHeteroscedasticLoss()

    loss = criterion(
        prediction,
        target
    )

    print(
        f"Prediction shape: "
        f"{prediction.shape}"
    )

    print(
        f"Target shape: "
        f"{target.shape}"
    )

    print(
        f"Loss: {loss.item():.6f}"
    )

    print(
        f"Loss is finite: "
        f"{torch.isfinite(loss).item()}"
    )

    # 测试反向传播
    loss.backward()

    print(
        f"Gradient shape: "
        f"{prediction.grad.shape}"
    )

    print(
        f"Gradient is finite: "
        f"{torch.isfinite(prediction.grad).all().item()}"
    )

    print("\nLoss test passed.")