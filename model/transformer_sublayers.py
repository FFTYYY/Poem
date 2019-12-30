import sys
import fastNLP
import torch as tc
from torch import nn as nn
import torch.nn.functional as F
import math
import pdb
import fitlog
import random

fitlog.commit(__file__)

def Attention(Q, K, V , q_mas , k_mas , att_mas = None):
	'''
		Q      : (bs,h,nq,dk)
		K,V    : (bs,h,nk,dk)
		q_mas : (bs,1,nq,1)
		k_mas : (bs,1,nk,1)
	'''
	bs,h,nq,d = Q.size()
	bs,h,nk,d = K.size()

	y = tc.matmul(Q , K.transpose(-1, -2)) # (bs,h,nq,nk)

	mas = q_mas * k_mas.transpose(-1, -2)  # (bs,1,nq,nk)
	if att_mas is not None:
		mas = mas * att_mas.view(bs,1,nq,nk)

	y = y + ((1-mas).float() * -10000)

	y = F.softmax(y / (d**0.5) , dim = -1) * mas
	y = y.matmul(V) * q_mas

	return y

class MultiHeadAttention(nn.Module):
	def __init__(self , h = 4 , d_model = 512 , drop_p = 0.0):
		super().__init__()

		self.WQ = nn.Linear(d_model, d_model, bias = False)
		self.WK = nn.Linear(d_model, d_model, bias = False)
		self.WV = nn.Linear(d_model, d_model, bias = False)

		self.WO = nn.Linear(d_model, d_model, bias = False)

		self.drop = nn.Dropout(drop_p)

		#-----hyper params-----

		self.dk = d_model // h
		self.h = h
		self.d_model = d_model

		self.reset_params()

	def reset_params(self):
		#nn.init.xavier_normal_(self.WQ.weight.data , gain = 1.0)
		#nn.init.xavier_normal_(self.WK.weight.data , gain = 1.0)
		#nn.init.xavier_normal_(self.WV.weight.data , gain = 1.0)
		#nn.init.xavier_normal_(self.WO.weight.data , gain = 1.0)
		pass

	def forward(self , Q , K , V , q_mas , k_mas = None , att_mas = None):
		'''
			Q: bs , n , d
			mas : bs , n , 1
			mas_att : bs , n , n
		'''
		#pdb.set_trace()

		#pdb.set_trace()
		bs , nq , d = Q.size()
		bs , nk , d = K.size()
		h = self.h
		q_mas = q_mas.view(bs,1,nq,1)		#(bs,1,n,1)
		if k_mas is None: 
			k_mas = q_mas
		else:
			k_mas = k_mas.view(bs,1,nk,1)

		Q = q_mas * self.WQ(Q).view(bs,nq,h,self.dk).transpose(1,2)	#(bs,h,n,dk)
		K = k_mas * self.WK(K).view(bs,nk,h,self.dk).transpose(1,2)	#(bs,h,n,dk)
		V = k_mas * self.WV(V).view(bs,nk,h,self.dk).transpose(1,2)	#(bs,h,n,dk)

		y = Attention(Q , K , V , q_mas , k_mas , att_mas)

		y = y.view(bs,h,nq,self.dk).transpose(1,2).contiguous().view(bs,nq,d)
		y = self.WO(y) * q_mas.view(bs,nq,1)

		y = self.drop(y)

		return y


class FFN(nn.Module):
	def __init__(self, d_model = 512 , d_hid = 512 , drop_p = 0.0):
		super().__init__()

		self.d_hid = d_hid
		self.L1 = nn.Linear(d_model , d_hid , bias = True)
		self.L2 = nn.Linear(d_hid , d_model , bias = True)
		self.drop = nn.Dropout(drop_p)		
		self.reset_params()

	def reset_params(self):
		#nn.init.xavier_normal_(self.L1.weight.data)
		#nn.init.xavier_normal_(self.L2.weight.data)
		#self.L1.bias.data.fill_(0)
		#self.L2.bias.data.fill_(0)
		pass

	def forward(self , x , mas):
		x = self.drop(F.relu(self.L1(x)))
		x = self.L2(x)
		x = x * mas
		return x

class Decoder_Layer(nn.Module):
	def __init__(self , d_model = 512 , d_hid = 512 , h = 4 , drop_p = 0.2 , n_extra = 0):
		super().__init__()

		self.self_att  = MultiHeadAttention(h = h , d_model = d_model , drop_p = drop_p)
		self.layernorm_1 = nn.LayerNorm([d_model])

		self.extra_layers = nn.ModuleList([
			MultiHeadAttention(h = h , d_model = d_model , drop_p = drop_p)
			for _ in range(n_extra)
		])
		self.layernorm_3 = nn.LayerNorm([d_model])

		self.ffn = FFN(d_model = d_model , d_hid = d_hid , drop_p = drop_p)
		self.layernorm_2 = nn.LayerNorm([d_model])

		#----- hyper params -----
		self.n_extra = n_extra

	def reset_params(self):
		self.self_att.reset_params()
		self.ffn.reset_params()

	def forward(self, x , seq_mas , att_mas = None , select_seqs = None):
		'''
			select_seqs = [
				[ ex_emb: (bsz , ex_len , d_model) , ex_mask: (bsz , ex_len) ]
			]

		'''

		bsz , slen , d = x.size()

		out1 = self.self_att(x , x , x , seq_mas , att_mas = att_mas)
		x = self.layernorm_1(x + out1)
		x *= seq_mas

		if self.n_extra > 0:
			out3 = 0
			for i , (ex_emb , ex_mask) in enumerate(select_seqs):
				
				now_y = self.extra_layers[i](
					x , ex_emb , ex_emb , 
					seq_mas , ex_mask , None , 
				)

				out3 = out3 + now_y

			out3 = out3 * seq_mas
			x = self.layernorm_3(x + out3)
			x *= seq_mas

		out2 = self.ffn(x , seq_mas)
		x = self.layernorm_2(x + out2)
		x *= seq_mas

		return x


class Decoder(nn.Module):
	def __init__(self , num_layers = 4 , d_model = 500 , d_hid = 1024 , h = 5 , drop_p = 0.1 , n_extra = 0):
		super().__init__()

		self.pos_emb = nn.Parameter(tc.zeros(1024 , d_model))

		self.dec_layers = nn.ModuleList([
			Decoder_Layer(d_model = d_model , d_hid = d_hid , h = h , drop_p = drop_p , n_extra = n_extra) 
			for _ in range(num_layers)
		])

		self.reset_params()

		#-----hyper params-----
		self.d_model = d_model
		self.num_layers = num_layers

	def reset_params(self):
		for x in self.dec_layers:
			x.reset_params()
		nn.init.normal_( self.pos_emb.data , 0 , 0.01)

	def forward(self , x , seq_mas , att_mas = None , select_seqs = None):
		'''
			x : (bs , n , emb_siz)
		'''
		bs , n , d = x.size()

		seq_mas = seq_mas.view(bs , n , 1).float()
		if att_mas is not None:
			att_mas = att_mas.view(bs , n , n).float()
		if att_mas is None:
			att_mas = 1
		att_mas = att_mas * tc.tril(x.new_ones(bs,n,n))

		x = x + self.pos_emb[:n,:].view(1,n,d)

		for i in range(self.num_layers):
			x = self.dec_layers[i](x  ,seq_mas , att_mas , select_seqs = select_seqs) #(bs , len , d_model)

		return x

class Encoder(nn.Module):
	def __init__(self , num_layers = 4 , d_model = 500 , d_hid = 1024 , h = 5 , drop_p = 0.1):
		super().__init__()

		self.pos_emb = nn.Parameter(tc.zeros(1024 , d_model))

		self.dec_layers = nn.ModuleList([
			Decoder_Layer(d_model = d_model , d_hid = d_hid , h = h , drop_p = drop_p , n_extra = 0) 
			for _ in range(num_layers)
		])

		self.reset_params()

		#-----hyper params-----
		self.d_model = d_model
		self.num_layers = num_layers

	def reset_params(self):
		for x in self.dec_layers:
			x.reset_params()
		nn.init.normal_( self.pos_emb.data , 0 , 0.01)
		#print ("parameters reset")

	def forward(self , x , seq_mas , att_mas = None):
		'''
			x : (bs , n , emb_siz)
		'''
		bs , n , d = x.size()

		seq_mas = seq_mas.view(bs , n , 1).float()
		if att_mas is not None:
			att_mas = att_mas.view(bs , n , n).float()

		x = x + self.pos_emb[:n,:].view(1,n,d)

		for i in range(self.num_layers):
			x = self.dec_layers[i](x  ,seq_mas , att_mas) #(bs , len , d_model)

		return x

