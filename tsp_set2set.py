import torch
from torch import nn
import pytorch_lightning as pl
from process_block import ProcessBlock
from ptr_network import PointerNet
import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import *

# inspired from ast0414 (https://github.com/ast0414/pointer-networks-pytorch/blob/master/train_sort.py)

class TspSet2SetNet(pl.LightningModule):
	"""
		This is the final network containing 3 layers:
		The embedding layer: a simple NN
		a Set2Set Aggregator based on the process block from the article
		a pointer network decoder - used for pointing the relevant indices in the data
	"""
	def __init__(self, input_dim, embedding_dim, num_processing_steps: int, max_seq_len, use_nll=True, lr=1e-3, wd=1e-5, num_layers=1, bidirectional=True, batch_first=True):
		super(TspSet2SetNet, self).__init__()
		self.input_dim = input_dim
		self.embedding_dim = embedding_dim
		self.num_processing_steps = num_processing_steps
		self.hidden_dim = embedding_dim * 2
		self.max_seq_len = max_seq_len
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.batch_first = batch_first
		self.optimizer_params = {'lr': lr, 'weight_decay': wd}
		self.tar_device = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.embedding_layer = nn.Linear(input_dim, embedding_dim)
		self.process_block = ProcessBlock(self.embedding_dim, self.num_processing_steps, self.max_seq_len, self.tar_device, self.num_layers, self.batch_first)
		self.write_block = PointerNet(self.embedding_dim, self.hidden_dim, self.tar_device, self.bidirectional, self.batch_first)
		self.use_nll = use_nll

	# for idx, (data, batch, label) in enumerate(train_loader):
	def training_step(self, batch, batch_idx):
		loss = 0
		data, batch_lens, label = batch
		embedded = self.embedding_layer(data)
		context_vector = self.process_block(embedded, batch_lens)
		log_pointer_score, argmax_pointer, mask = self.write_block(embedded, data, batch_lens, context_vector)

		unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
		if self.use_nll:
			loss = F.nll_loss(unrolled, label.view(-1), ignore_index=-1)
		else:
			loss_fn = torch.nn.CrossEntropyLoss()  #
			loss = loss_fn(unrolled, label.view(-1), ignore_index=-1)
		# loss = F.nll_loss(unrolled, label.view(-1), ignore_index=-1)
		acc = masked_accuracy(argmax_pointer, label, mask)
		self.log_dict({'train_acc': acc, 'train_loss': loss}, on_step=True, on_epoch=True)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
		return loss

	def validation_step(self, batch, batch_idx):
		data, batch_lens, label = batch
		embedded = self.embedding_layer(data)
		context_vector = self.process_block(embedded, batch_lens)
		log_pointer_score, argmax_pointer, mask = self.write_block(embedded, data, batch_lens, context_vector)

		unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
		loss = F.nll_loss(unrolled, label.view(-1), ignore_index=-1)
		acc = masked_accuracy(argmax_pointer, label, mask)
		self.log_dict({'val_acc': acc, 'val_loss': loss}, on_step=True, on_epoch=True)

	def configure_optimizers(self):
		optimizer = Adam(self.parameters(), **self.optimizer_params)
		return optimizer





