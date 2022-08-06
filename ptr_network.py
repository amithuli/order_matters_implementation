import torch
from torch import nn
from utils import *

# Inspired from ast0414 (https://github.com/ast0414/pointer-networks-pytorch/blob/master/model.py)

# Inspired from ast0414 (https://github.com/ast0414/pointer-networks-pytorch/blob/master/model.py)

class Attention(nn.Module):
	def __init__(self, hidden_size):
		super(Attention, self).__init__()
		self.hidden_size = hidden_size
		self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
		self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
		self.vt = nn.Linear(hidden_size, 1, bias=False)

	def forward(self, decoder_state, encoder_outputs, mask):
		# (batch_size, max_seq_len, hidden_size)
		encoder_transform = self.W1(encoder_outputs)

		# (batch_size, 1 (unsqueezed), hidden_size)
		decoder_transform = self.W2(decoder_state).unsqueeze(1)

		# 1st line of Eq.(3) in the paper
		# (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
		u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

		# softmax with only valid inputs, excluding zero padded parts
		# log-softmax for a better numerical stability
		log_score = masked_log_softmax(u_i, mask, dim=-1)

		return log_score


class Encoder(nn.Module):
	def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
		super(Encoder, self).__init__()

		self.batch_first = batch_first
		self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
						   batch_first=batch_first, bidirectional=bidirectional)

	def forward(self, embedded_inputs, input_lengths):
		# Pack padded batch of sequences for RNN module
		input_lengths = input_lengths.cpu()
		packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths, batch_first=self.batch_first)
		# Forward pass through RNN
		outputs, hidden = self.rnn(packed)
		# Unpack padding
		outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)
		# Return output and final hidden state
		return outputs, hidden

class PointerNet(nn.Module):
	def __init__(self, embedding_dim, hidden_size, device, bidirectional=True, batch_first=True):
		super(PointerNet, self).__init__()

		# Embedding dimension
		self.embedding_dim = embedding_dim
		# (Decoder) hidden size
		self.hidden_size = hidden_size
		# Bidirectional Encoder
		self.bidirectional = bidirectional
		self.num_directions = 2 if bidirectional else 1
		self.num_layers = 1
		self.batch_first = batch_first
		self.device = device

		# We use an embedding layer for more complicate application usages later, e.g., word sequences.
		self.encoder = Encoder(embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=self.num_layers,
							   bidirectional=bidirectional, batch_first=batch_first).to(self.device)
		self.decoding_rnn = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size).to(self.device)
		self.attn = Attention(hidden_size=hidden_size).to(self.device)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				if m.bias is not None:
					torch.nn.init.zeros_(m.bias)

	def forward(self, embedded, input_seq, input_lengths, context):

		if self.batch_first:
			batch_size = input_seq.size(0)
			max_seq_len = input_seq.size(1)
		else:
			batch_size = input_seq.size(1)
			max_seq_len = input_seq.size(0)

		# encoder_output => (batch_size, max_seq_len, hidden_size) if batch_first else (max_seq_len, batch_size, hidden_size)
		# hidden_size is usually set same as embedding size
		# encoder_hidden => (num_layers * num_directions, batch_size, hidden_size) for each of h_n and c_n
		encoder_outputs, encoder_hidden = self.encoder(embedded, input_lengths)

		if self.bidirectional:
			# Optionally, Sum bidirectional RNN outputs
			encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]

		encoder_h_n, encoder_c_n = encoder_hidden
		encoder_h_n = encoder_h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
		encoder_c_n = encoder_c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

		# Lets use zeros as an intial input for sorting example
		# decoder_input = encoder_outputs.new_zeros(torch.Size((batch_size, self.hidden_size)))
		decoder_input = context
		decoder_hidden = (encoder_h_n[-1, 0, :, :].squeeze(), encoder_c_n[-1, 0, :, :].squeeze())

		range_tensor = torch.arange(max_seq_len, device=input_lengths.device, dtype=input_lengths.dtype).expand(batch_size, max_seq_len, max_seq_len).to(self.device)
		each_len_tensor = input_lengths.view(-1, 1, 1).expand(batch_size, max_seq_len, max_seq_len)

		row_mask_tensor = (range_tensor < each_len_tensor)
		col_mask_tensor = row_mask_tensor.transpose(1, 2)
		mask_tensor = row_mask_tensor * col_mask_tensor

		pointer_log_scores = []
		pointer_argmaxs = []

		for i in range(max_seq_len):
			# We will simply mask out when calculating attention or max (and loss later)
			# not all input and hiddens, just for simplicity
			sub_mask = mask_tensor[:, i, :].float()

			# h, c: (batch_size, hidden_size)
			h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)

			# next hidden
			decoder_hidden = (h_i, c_i)

			# Get a pointer distribution over the encoder outputs using attention
			# (batch_size, max_seq_len)
			log_pointer_score = self.attn(h_i, encoder_outputs, sub_mask)
			pointer_log_scores.append(log_pointer_score)

			# Get the indices of maximum pointer
			_, masked_argmax = masked_max(log_pointer_score, sub_mask, dim=1, keepdim=True)

			pointer_argmaxs.append(masked_argmax)
			index_tensor = masked_argmax.unsqueeze(-1).expand(batch_size, 1, self.hidden_size)

			# (batch_size, hidden_size)
			decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)

		pointer_log_scores = torch.stack(pointer_log_scores, 1)
		pointer_argmaxs = torch.cat(pointer_argmaxs, 1)

		return pointer_log_scores, pointer_argmaxs, mask_tensor