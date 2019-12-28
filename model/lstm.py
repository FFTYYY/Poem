import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import pdb
import fitlog

fitlog.commit(__file__)

class LSTM(nn.Module):
	def __init__(self , input_size , hidden_size , num_layers , bidrect , dropout = 0.0 , pos_len = True , output_mode = None):
		'''
			output_mode: 
				None: return y and h
				"seq": return y (token encoder)
				"vec": return h (sequence encoder)

		'''
		super().__init__()

		if num_layers <= 1:
			dropout = 0.0
		
		self.rnn = nn.LSTM(input_size = input_size , hidden_size = hidden_size , 
			num_layers = num_layers , batch_first = True , dropout = dropout , 
			bidirectional = bidrect)

		self.number = (2 if bidrect else 1) * num_layers

		self.pos_len = pos_len
		self.output_mode = output_mode

		if self.output_mode == "seq":
			self.out_dim = (2 if bidrect else 1) * hidden_size
		elif self.output_mode == "vec":
			self.out_dim = self.number * hidden_size

	def forward(self , x , mask = None , lens = None):
		if self.pos_len:
			y , h = self.forward_for_pos_len(x , mask , lens)
		else:
			y , h = self.forward_for_zer_len(x , mask , lens)

		if self.output_mode is None:
			return y , h
		if self.output_mode == "seq":
			return y
		if self.output_mode == "vec":
			return h
		raise Exception("bad output_mode")

	def forward_for_pos_len(self , x , mask = None, lens = None):
		'''这个函数只处理长度>0的输入

			x : (bsz , sl , is)
			mask : (bsz , sl) 
			lens : (bsz)
		'''
		assert mask is not None or lens is not None
		if lens is None:
			lens = (mask).long().sum(dim = 1)
		lens , idx_sort = tc.sort(lens , descending = True)
		_ , idx_unsort = tc.sort(idx_sort)

		x = x[idx_sort]
		
		x = nn.utils.rnn.pack_padded_sequence(x , lens , batch_first = True)
		self.rnn.flatten_parameters()
		y , (h , c) = self.rnn(x)
		y , lens = nn.utils.rnn.pad_packed_sequence(y , batch_first = True)

		h = h.transpose(0,1).contiguous() #make batch size first

		y = y[idx_unsort]							#(bsz , seq_len , bid * hid_size)
		h = h[idx_unsort].view(h.size(0),-1)		#(bsz , number , bid * hid_size)

		return y , h

	def forward_for_zer_len(self , x , mask = None , lens = None):
		'''这个函数可以处理输入长度=0的情况

			x : (bs , sl , is)
			mask : (bs , sl) 
			lens : (bs)

			return:
				y: (bsz , seq_len , bid * hid_size)
				h: (bsz , bid , hid_size)
		'''
		assert mask is not None or lens is not None
		if lens is None:
			lens = (mask).long().sum(dim = 1)

		bsz , seq_len , d_model = x.size()

		tot_mask = (lens != 0)
		good_range = tc.arange(bsz).to(x.device).masked_select( tot_mask)	# 那些有正长度的下标
		bad_range  = tc.arange(bsz).to(x.device).masked_select(~tot_mask) 	# 那些长度为0的下标

		y = x.masked_select(tot_mask.view(bsz,1,1)).view(-1,seq_len,d_model)
		y,h = self.forward_for_pos_len(y , lens = lens.masked_select(tot_mask))

		idx = tc.cat([good_range , bad_range] , dim = -1) # 此时 idx[k] 表示y[k]对应y[idx[k]]
		idx = tc.sort(idx)[1] # 这个idx是恢复顺序的selector

		y = tc.cat([y , y.new_zeros(bad_range.size(0) , y.size(1) , y.size(2))] , dim = 0)[idx]
		h = tc.cat([h , h.new_zeros(bad_range.size(0) , h.size(1) , h.size(2))] , dim = 0)[idx]

		return y , h