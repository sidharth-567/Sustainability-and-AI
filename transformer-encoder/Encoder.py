import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import r2_score
import math


class TimeSeriesTransformer(pl.LightningModule):
    def __init__(
            self,
            input_dim=51,  # 3*17
            output_dim=39,  # 3*13
            d_model=256,
            nhead=8,
            num_layers=6,
            dropout=0.1,
            weight_decay=0.01,
            max_lr=1e-3,
            min_lr=1e-5,
            warmup_epochs=5,
            max_epochs=100
    ):
        super().__init__()
        self.save_hyperparameters()

        # Input projection MLP
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

        # Learned positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))

        # Pre-LayerNorm Transformer Encoder
        encoder_layer = PreLNTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)

        # Metrics for testing
        self.test_mse = []
        self.test_outputs = []
        self.test_targets = []

        # Warmup related
        self.warmup_epochs = warmup_epochs
        self.automatic_optimization = False  # We'll handle optimization manually

    def forward(self, x):
        batch_size = x.shape[0]

        # Reshape input from [B, 3, 17] to [B, 3*17]
        x = x.reshape(batch_size, -1)

        # Project input to d_model dimension
        x = self.input_projection(x)  # [B, d_model]

        # Add positional encoding
        x = x.unsqueeze(1)  # [B, 1, d_model]
        x = x + self.pos_encoding

        # Transformer encoder
        x = x.transpose(0, 1)  # [1, B, d_model]
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # [B, 1, d_model]

        # Project to output dimension
        x = x.squeeze(1)  # [B, d_model]
        x = self.output_projection(x)  # [B, 3*13]

        # Reshape to final output shape
        x = x.reshape(batch_size, 3, 13)

        return x

    def get_lr_factor(self, current_epoch):
        """Calculate learning rate factor for warm-up and cosine decay"""
        if current_epoch < self.warmup_epochs:
            # Linear warmup
            return current_epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (current_epoch - self.warmup_epochs) / (self.hparams.max_epochs - self.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        # Calculate current learning rate
        current_epoch = self.current_epoch + batch_idx / 1000  # Smooth approximation
        lr_factor = self.get_lr_factor(current_epoch)
        lr = self.hparams.min_lr + (self.hparams.max_lr - self.hparams.min_lr) * lr_factor

        # Update learning rate
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        # Forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        # Backward pass
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('lr', lr, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.test_mse.append(loss.item())
        self.test_outputs.append(y_hat.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        return loss

    def on_test_epoch_end(self):
        avg_mse = sum(self.test_mse) / len(self.test_mse)

        y_pred = torch.cat(self.test_outputs, dim=0).numpy()
        y_true = torch.cat(self.test_targets, dim=0).numpy()

        y_pred_2d = y_pred.reshape(-1, y_pred.shape[-1])
        y_true_2d = y_true.reshape(-1, y_true.shape[-1])
        r2 = r2_score(y_true_2d, y_pred_2d)

        print(f"\nTest Results:")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"R² Score: {r2:.6f}")

        self.test_mse = []
        self.test_outputs = []
        self.test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.max_lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

class PreLNTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        # Pre-LN for attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2,
                                 attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask,
                                 is_causal=is_causal)
        src = src + self.dropout(src2)

        # Pre-LN for feedforward
        src2 = self.norm2(src)
        src2 = self.ff(src2)
        src = src + src2

        return src
