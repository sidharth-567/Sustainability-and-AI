import torch
import pytorch_lightning as pl
from data import SAIDataset,SAILightningDataModule
from LSTM import TimeSeriesLSTM
import wandb
from pytorch_lightning.loggers import WandbLogger
import os
from dotenv import load_dotenv
load_dotenv()
from pytorch_lightning.callbacks import EarlyStopping, Callback, ModelCheckpoint
import torch.nn.init as init
import torch.nn as nn


if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    torch.set_float32_matmul_precision('medium')
    freeze_support()

    dataloader = SAILightningDataModule(data_dir="../cleaned_data_6.csv", batch_size=32,
                                        val_split=0.2, test_split=0.2, seed=6)

    model = TimeSeriesLSTM(input_size=17,output_size=13, hidden_size=512, num_layers=4, dropout=0.3, learning_rate=1e-3, min_lr=1e-6,
                           total_epochs=100, warmup_epochs=5, gradient_clip_val=1.0, weight_decay=0.01)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="pre-trained/",
        filename="MLP-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )


    class GlorotInitializationCallback(pl.Callback):
        def on_fit_start(self, trainer, pl_module):
            """
            This method is called when fit starts. It iterates through all the layers
            of the model and applies Glorot initialization to linear layers.
            """
            for module in pl_module.modules():
                if isinstance(module, nn.Linear):
                    init.xavier_uniform_(module.weight)  # Apply Glorot uniform initialization
                    if module.bias is not None:
                        init.zeros_(module.bias)  # Set biases to 0


    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min'
    )

    trainer = pl.Trainer(max_epochs=100,
                         accelerator='gpu',
                         devices=1,
                         precision='16-mixed',
                         callbacks=[checkpoint_callback, early_stopping, GlorotInitializationCallback()],
                         check_val_every_n_epoch=1,
                         log_every_n_steps=5,
                         enable_progress_bar=True)

    trainer.fit(model, datamodule=dataloader)
    trainer.test(model, datamodule=dataloader)
