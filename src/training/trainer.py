from pathlib import Path

import torch


class Trainer:
    """
    SDB model trainer.

    负责：
    1. 单个 epoch 的训练
    2. 验证集评估
    3. 优化器更新
    4. 保存最佳模型
    5. 记录训练历史
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device="auto",
        checkpoint_dir="checkpoints",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        # 自动选择计算设备
        if device == "auto":
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

        print(f"Training device: {self.device}")

        if self.device.type == "cuda":
            print(
                f"GPU: {torch.cuda.get_device_name(0)}"
            )

    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.

        Parameters
        ----------
        train_loader : DataLoader

        Returns
        -------
        float
            Average training loss.
        """

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for X, y in train_loader:

            X = X.to(self.device)
            y = y.to(self.device)

            # 清空上一轮梯度
            self.optimizer.zero_grad()

            # 前向传播
            prediction = self.model(X)

            # 计算 loss
            loss = self.criterion(
                prediction,
                y
            )

            # 反向传播
            loss.backward()

            # 更新参数
            self.optimizer.step()

            batch_size = X.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        average_loss = (
            total_loss / total_samples
        )

        return average_loss

    @torch.no_grad()
    def validate(self, val_loader):
        """
        Evaluate the model on validation data.

        Parameters
        ----------
        val_loader : DataLoader

        Returns
        -------
        float
            Average validation loss.
        """

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for X, y in val_loader:

            X = X.to(self.device)
            y = y.to(self.device)

            prediction = self.model(X)

            loss = self.criterion(
                prediction,
                y
            )

            batch_size = X.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        average_loss = (
            total_loss / total_samples
        )

        return average_loss

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
    ):
        """
        Train the model for multiple epochs.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        epochs : int

        Returns
        -------
        dict
            Training history.
        """

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):

            train_loss = self.train_epoch(
                train_loader
            )

            val_loss = self.validate(
                val_loader
            )

            self.history["train_loss"].append(
                train_loss
            )

            self.history["val_loss"].append(
                val_loss
            )

            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

            # 保存验证集表现最好的模型
            if val_loss < best_val_loss:

                best_val_loss = val_loss

                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_loss,
                    filename="best_model.pth"
                )

        print(
            f"\nBest validation loss: "
            f"{best_val_loss:.6f}"
        )

        return self.history

    def save_checkpoint(
        self,
        epoch,
        val_loss,
        filename="best_model.pth",
    ):
        """
        Save model checkpoint.
        """

        checkpoint_path = (
            self.checkpoint_dir / filename
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    self.model.state_dict(),
                "optimizer_state_dict":
                    self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path
        )

        print(
            f"Saved checkpoint: "
            f"{checkpoint_path}"
        )