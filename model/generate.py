import sys
import fastNLP
import torch as tc
from torch import nn as nn
import torch.nn.functional as F
import math
import pdb
import fitlog
import random
import pypinyin

fitlog.commit(__file__)

def manually_modify(pre_tok , prob , vocab):
	'''
		降低重复出现的词的概率
	'''
	skip_tokens = "。，"
	for x in pre_tok:
		if x in [vocab.to_index(w) for w in skip_tokens]:
			continue 
		prob[x] *= 0.1
	return prob

def endswith_ng(s):
	return s.endswith("ing") or s.endswith("ong") or s.endswith("eng")
def endswith_n(s):
	return s.endswith("in") or s.endswith("un") or s.endswith("en")

def manually_select(pre_tok , candidate , vocab):

	pre_tok = pre_tok[1:] #去掉<SOS>
	#----- 找到首个句号 '。' -----
	pos = -1
	for i in range(len(pre_tok)):
		if vocab.to_word(pre_tok[i]) in ["！" , "。"]:
			pos = i
			break
	if pos <= 0: #没找到句号
		return candidate

	#----- 句末检测 ------
	if len(pre_tok) % (pos+1) != (pos-1):
		return candidate

	#----- 押韵筛选 ------
	vowol = pypinyin.pinyin(vocab.to_word(pre_tok[pos-1]) , style = pypinyin.STYLE_FINALS)[0][0]
	ret = []
	for x in candidate:
		x_vowol = pypinyin.pinyin(vocab.to_word(x) , style = pypinyin.STYLE_FINALS)[0][0]
		if x_vowol == vowol or (
			endswith_ng(x_vowol) 	and endswith_ng(vowol)) 	or (
			endswith_n(x_vowol) 	and endswith_n(vowol)) 		or (
			x_vowol.endswith("ao") 	and vowol.endswith("ao")) 	or (
			x_vowol.endswith("an") 	and vowol.endswith("an"))	:
			ret.append(x)
	return ret




def generate(net , x , norm = True , beam_size = 15 , max_len = 96):
	'''
		x: (seq_len)
	'''

	x = x.view(1,x.size(0))
	sos_id = net.vocab.word2idx["<SOS>"]
	eos_id = net.vocab.word2idx["<EOS>"]

	ys = [{
		"toks":[sos_id],
		"log_prob" : 0.,
	}]

	for i in range(max_len):

		if len(ys) <= 0:
			break

		new_ys = []

		flag = False
		for y in ys:
			toks,log_prob = y["toks"],y["log_prob"]
			if len(toks) > 0 and toks[-1] == eos_id:
				new_ys.append(y)
				continue
			
			flag = True #不全是eos

			with tc.no_grad():
				now_y = tc.LongTensor(toks).view(1 , len(toks)).to(x.device)
				new_y = net(x , now_y)[0 , -1] # ( len(vovab) )
				new_y = tc.softmax(new_y , dim = -1)

			new_y = manually_modify(y["toks"] , new_y , net.vocab)

			#-----------beam--------------

			_beam_size = beam_size
			top_idx = []
			while len(top_idx) <= 0:
				top_idx = [int(x) for x in new_y.topk(_beam_size)[1]]
				top_idx = manually_select(y["toks"] , top_idx , net.vocab)
				_beam_size = _beam_size * 2

			for idx in top_idx:
				n_log_prob = log_prob + tc.log(new_y[idx])

				n_toks = toks + [idx]
				new_ys.append({
					"toks" : n_toks,
					"log_prob" : n_log_prob,
				})

		if norm:
			new_ys.sort(key = lambda x:-(x["log_prob"] / len(x["toks"])))
		else:
			new_ys.sort(key = lambda x:-(x["log_prob"]))

		ys = new_ys[:beam_size]
		if not flag:
			break

	return ys[0]["toks"]
