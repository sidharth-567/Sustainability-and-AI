import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW


class TimeSeriesLSTM(pl.LightningModule):
    def __init__(
            self,
            input_size: int = 17,
            output_size: int = 13,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            learning_rate: float = 1e-3,
            min_lr: float = 1e-6,
            total_epochs: int = 100,
            warmup_epochs: int = 5,
            gradient_clip_val: float = 1.0,
            weight_decay: float = 0.01,
    ):
        """
        Args:
            input_size: Number of input features at each timestep
            output_size: Number of output features at each timestep
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Maximum learning rate after warmup
            min_lr: Minimum learning rate at the end of cosine schedule
            total_epochs: Total number of training epochs
            warmup_epochs: Number of warmup epochs
            gradient_clip_val: Value for gradient clipping
            weight_decay: Weight decay for AdamW optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        # Model parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Training parameters
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.gradient_clip_val = gradient_clip_val
        self.weight_decay = weight_decay

        # Initialize lists to store predictions and targets for test set
        self.test_preds = []
        self.test_targets = []

        # Model layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # Batch normalization after LSTM
        self.batch_norm = nn.BatchNorm1d(3)  # normalize across timesteps

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Final linear layer (note: *2 because LSTM is bidirectional)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x shape: [batch_size, timesteps, input_features]
        batch_size = x.size(0)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch_size, timesteps, hidden_size * 2]

        # Apply batch normalization
        lstm_out = self.batch_norm(lstm_out)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Apply linear layer to each timestep
        output = self.linear(lstm_out)
        # output shape: [batch_size, timesteps, output_features]

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)

        # Collect predictions and targets
        if not hasattr(self, 'all_preds'):
            self.all_preds = []
            self.all_targets = []
        self.all_preds.append(y_hat.detach())
        self.all_targets.append(y.detach())

        return {'test_loss': loss}

    def on_test_end(self):
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.all_preds, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)

        # Calculate overall loss
        overall_loss = nn.MSELoss()(all_preds, all_targets)
        print('overall_test_loss', overall_loss)

        # Calculate R² (Coefficient of Determination)
        residuals = all_preds - all_targets
        ss_res = torch.sum(residuals ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)

        if ss_tot.item() == 0:
            r2 = torch.tensor(0.0)
        else:
            r2 = 1 - (ss_res / ss_tot)

        print('r2_score', r2)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.total_epochs - self.warmup_epochs,
            eta_min=self.min_lr
        )

        # Custom learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return epoch / self.warmup_epochs
            return 1.0  # After warmup, let cosine annealing take over

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )

        # Scheduler dict for PyTorch Lightning
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
        }

        return [optimizer], [scheduler_config]