from typing import Dict, Tuple, Any
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR


class MLP(pl.LightningModule):
    """
    Input args:
    dropout (float): Dropout rate between 0 and 1.
    architecture (Dict[str, Tuple[int, int]]): Must be of the format {layer_name: (input_dimension, output_dimension)}
    """

    def __init__(self,architecture: Dict[str, Tuple[int, int]], dropout: float=0.3, learning_rate:float = 1e-3):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.lr = learning_rate

        # "architecture": {"l1":(18,128),"l2":(128,512),"l3":(512,128),"output":(128,18)},
        # Build the layers
        for layer_name, (input_dim, output_dim) in architecture.items():
            self.layers.append(nn.Linear(input_dim, output_dim))
            if layer_name != 'output':
                self.layers.append(nn.BatchNorm1d(output_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(self.dropout))

    def forward(self, x):
        # Reshape from [batch_size, 3, 17] to [batch_size * 3, 17]
        batch_size = x.size(0)
        x = x.view(batch_size * 3, 17)  # Flatten the 3 dimension

        # Pass through the layers
        for layer in self.layers:
            x = layer(x)

        # Reshape back to [batch_size, 3, 18]
        x = x.view(batch_size, 3, 13)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}

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
        # Define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Warm-up function: Linearly increase learning rate during warm-up phase
        def lr_lambda_warmup(epoch):
            warmup_epochs = 5  # Number of warm-up epochs
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 1.0

        # Define warm-up scheduler
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)

        # Define CosineAnnealingWarmRestarts scheduler
        cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # Combine warm-up and cosine annealing in a single scheduler
        scheduler = {
            "scheduler": warmup_scheduler,  # Start with warm-up
            "interval": "epoch",  # Warm-up works per epoch
            "frequency": 1,
            "reduce_on_plateau": False,  # Not using ReduceLROnPlateau
            "name": "warmup_scheduler",
        }

        # Use manual step to switch to cosine annealing after warm-up
        def combined_scheduler(epoch):
            if epoch < 5:  # If within warm-up period
                warmup_scheduler.step()
            else:  # Switch to CosineAnnealingWarmRestarts
                cosine_scheduler.step(epoch - 5)

        return [optimizer], [scheduler]
