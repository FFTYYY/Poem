import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import pdb
import fitlog
from .transformer_sublayers import Encoder , Decoder
from fastNLP.embeddings.bert_embedding import BertEmbedding
from fastNLP.embeddings.static_embedding import StaticEmbedding

fitlog.commit(__file__)

class Model(nn.Module):
	def __init__(self , vocab , logger , num_layers = 6, d_hid = 1024 , h = 8 , d_model = 512 , dropout = 0.0):
		super().__init__()

		#self.b_embedding = BertEmbedding(vocab, model_dir_or_name='cn-wwm', requires_grad = False, layers='4,-2,-1')
		#self.b_emb_outer = nn.Linear(self.b_embedding.embed_size , d_model)
		self.s_embedding = StaticEmbedding(vocab , "cn-sgns-literature-word" , requires_grad = True)
		self.s_emb_outer = nn.Linear(self.s_embedding.embed_size , d_model)
		#self.r_embedding = nn.Embedding(len(vocab) , d_model , padding_idx = vocab.to_index("<pad>"))
		#self.r_emb_outer = nn.Linear(d_model , d_model)


		self.encoder 	= Encoder(
			num_layers = num_layers , d_model = d_model , d_hid = d_hid , h = h , drop_p = dropout
		)
		self.x_outer 	= nn.Linear(d_model , d_model)

		self.decoder 	= Decoder(
			num_layers = num_layers , d_model = d_model , d_hid = d_hid , h = h , 
			drop_p = dropout , n_extra = 1
		)
		self.y_outer 	= nn.Linear(d_model , d_model)
		self.output_ln 	= nn.Linear(d_model , len(vocab))

		#----- hyper params -----
		self.vocab = vocab
		self.d_model = d_model
		self.logger = logger

		#----- write sth -----
		#self.logger.log("用了Bert Embedding")

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

		x_mask = (x != 0)
		y_mask = (y != 0)

		x = F.relu(self.s_emb_outer(self.s_embedding(x)))
		y = F.relu(self.s_emb_outer(self.s_embedding(y)))

		x = self.encoder(x , seq_mas = x_mask) 
		x = F.relu(self.x_outer(x)) # (bsz , x_len , d_model)

		y = self.decoder(y , seq_mas = y_mask , select_seqs = [[x , x_mask]]) #(bsz , y_len , out_dim)
		y = F.relu(self.y_outer(y))
		y = self.output_ln(y)
		
		return y



