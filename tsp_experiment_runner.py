# Running Experiments using the following function
import torch.cuda

from dataset_creation import sparse_seq_collate_fn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tsp_dataset_creation import TSPDataset
from tsp_set2set import TspSet2SetNet


def run_experiment_tsp( embedding_size, input_dim=2, use_nll=True, num_processing_steps=2,seq_max_len=5, num_samples=100000, num_layers=1, max_epochs=1000, batch_size=256):
    '''Running an experiment with following parameters'''

    train_set = TSPDataset(num_samples, seq_max_len)
    train_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2, collate_fn=sparse_seq_collate_fn)

    validation_set = TSPDataset(num_samples//4, seq_max_len)
    validation_loader = DataLoader(validation_set,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2, collate_fn=sparse_seq_collate_fn)

    net = TspSet2SetNet(input_dim, embedding_size, num_processing_steps, seq_max_len, num_layers=num_layers)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(max_epochs=max_epochs,  check_val_every_n_epoch=10, accelerator=accelerator, callbacks=[pl.callbacks.EarlyStopping('val_loss', patience=8)])
    trainer.fit(model=net, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    return net