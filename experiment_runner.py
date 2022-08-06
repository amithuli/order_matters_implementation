import torch.cuda

from set_2_set_net import Set2SetNet
from dataset_creation import IntegerSortDataset, sparse_seq_collate_fn
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def run_experiment(input_dim, embedding_size, seq_min_len, seq_max_len, num_samples, num_processing_steps, num_layers=1, max_epochs=10000, batch_size=64):
    '''Running an experiment with following parameters'''
    train_set = IntegerSortDataset(num_samples=num_samples, high=input_dim, min_len=seq_min_len, max_len=seq_max_len, seed=1)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=sparse_seq_collate_fn)

    validation_set = IntegerSortDataset(num_samples=num_samples//4, high=input_dim, min_len=seq_min_len, max_len=seq_max_len, seed=1)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=sparse_seq_collate_fn)

    net = Set2SetNet(input_dim, embedding_size, num_processing_steps, seq_max_len, num_layers=num_layers)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(max_epochs=40, accelerator=accelerator, callbacks=[pl.callbacks.EarlyStopping('val_loss'), pl.callbacks.RichModelSummary(), pl.callbacks.RichProgressBar()])
    trainer.fit(model=net, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    return net
