import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from .lstm import LSTM
import pdb
import fitlog

fitlog.commit(__file__)

class Model(nn.Module):
	def __init__(self , vocab , d_model = 512 , dropout = 0.0):
		super().__init__()

		self.vocab = vocab

		self.embedding 	= nn.Embedding(len(vocab) , d_model , padding_idx = 0)
		self.ln 		= nn.Linear(d_model , d_model)
		self.output_ln 	= nn.Linear(d_model , len(vocab))

	def forward(self , x , y):
		'''
		params:
			x: (bsz , x_len)
			y_inpt: (bsz , y_len)

		return:
			(bsz , y_len , len(vocab))

		'''
		bsz , x_len = x.size()
		bsz , y_len = y.size()


		y = self.embedding(y)
		y = F.relu(self.ln(y))
		y = self.output_ln(y)

		return y



