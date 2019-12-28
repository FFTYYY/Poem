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

		self.embedding = nn.Embedding(len(vocab) , d_model , padding_idx = 0)

		self.x_encoder 	= LSTM(d_model , d_model , 2 , True , dropout , output_mode = "vec")
		self.x_outer 	= nn.Linear(self.x_encoder.out_dim , d_model)

		self.y_inputer 	= nn.Linear(d_model , d_model)
		self.decoder 	= LSTM(d_model , d_model , 2 , True , dropout , output_mode = "seq")
		self.output_ln 	= nn.Linear(self.decoder.out_dim , len(vocab))

		#----- hyper params -----
		self.vocab = vocab
		self.d_model = d_model

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
		d_model = self.d_model

		x_mask = (x != 0)
		y_mask = (y != 0)

		x = self.embedding(x)
		y = self.embedding(y)

		x = self.x_encoder(x , mask = x_mask) 
		x = F.relu(self.x_outer(x)) # (bsz , d_model)

		y = y + x.view(bsz , 1 , d_model)
		y = F.relu(self.y_inputer(y))
		y = self.decoder(y , mask = y_mask) #(bsz , y_len , out_dim)
		y = self.output_ln(y)
		
		return y



