import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
# Inspired from the phcavelar (https://github.com/phcavelar/graph-odenet/blob/cba1224c041e53ea221e31bf9103ef950b8bd460/QC/set2set.py)

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class ProcessBlock(nn.Module):
	"""
	Args:
	   embedding_size (int): Size of each input sample.
	   processing_steps (int): Number of iterations :math:`T`.
	   **kwargs (optional): Additional arguments of :class:`torch.nn.LSTM`.
	   """

	def __init__(self, embedding_size: int, processing_steps: int, max_seq_len, device, num_layers=1, batch_first=True,
				 **kwargs):
		super(ProcessBlock, self).__init__()
		self.in_channels = embedding_size
		self.out_channels = 2 * embedding_size
		self.processing_steps = processing_steps
		self.num_layers = num_layers
		self.lstm = torch.nn.LSTM(self.out_channels, embedding_size, **kwargs).to(device)
		self.batch_first = batch_first
		self.max_seq_len = max_seq_len
		self.reset_parameters()
		self.device = device

	def reset_parameters(self):
		self.lstm.reset_parameters()

	def forward(self, x, batch):
		# batch_size = batch.max().item() + 1
		batch_size = x.size()[0]
		num_elems = x.size()[1]

		# print(x.size())
		h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)).to(self.device),
			 x.new_zeros((self.num_layers, batch_size, self.in_channels)).to(self.device))
		q_star = x.new_zeros(batch_size, self.out_channels).to(self.device)
		# max_len_of_inputs
		mask_with = torch.arange(x.size()[1])
		for i in range(self.processing_steps):
			q, h = self.lstm(q_star.unsqueeze(0), h)
			q = q.view(batch_size, self.in_channels)
			# pdb.set_trace()
			e = torch.einsum('bij, kbj->bi', x, q.unsqueeze(0))
			# Softmax
			a = torch.zeros((batch_size, num_elems), dtype=x.dtype, device=x.device).to(self.device)
			for i in range(batch_size):
				mask = torch.where(mask_with < batch[i].item(), 1, 0).to(self.device)
				# masking the softmax per length size
				softmaxed_elements = F.softmax(e[i].masked_fill((1 - mask).bool(), float('-inf')), dim=0)
				# a[mask] += softmaxed_elements
				a[i] += softmaxed_elements

			# end for

			a = a.unsqueeze(0)
			r = torch.einsum('kij, bki -> kj', x, a)
			q_star = torch.cat([q, r], dim=-1)
		# end for

		return q_star

	def __repr__(self) -> str:
		return (f'{self.__class__.__name__}({self.in_channels}, '
				f'{self.out_channels})')

