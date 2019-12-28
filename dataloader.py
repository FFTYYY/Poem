from fastNLP import Vocabulary , DataSet , Instance
from config import C
import os
import fitlog
import pdb
import pickle
from utils import logger
import random
from opencc import OpenCC

chinese_converter = OpenCC("t2s")

fitlog.commit(__file__)

def is_chinese_char(c):
	#return '\u4e00' <= c and c <= '\u9fa5'
	return len(bytes(c , encoding = "utf-8")) == 3

def process_chinese_sent(s):
	s = s.lower()
	ret = []
	now = ""
	for c in s:

		if not c.strip():			# white spaces
			if now: 
				ret.append(now)
				now = ""
			continue
		elif is_chinese_char(c):	# chinese char
			if now: 
				ret.append(now)
				now = ""
			ret.append(c)

		else:						# non-chinese char
			now += c
	if now:
		ret.append(now)
	return ret

def chinese_tokenizer(s):
	s = chinese_converter.convert(s) #一律转化为繁体
	return ["<SOS>"] + process_chinese_sent(s.strip()) + ["<EOS>"]

vocab = Vocabulary(min_freq = 2)

def load(path):

	data = DataSet()
	_data = []

	with open(path , "r" , encoding = "utf-8") as fil:
		fil.readline()

		for line in fil:
			try:
				tradi , verna = line.strip().split("\t")
			except ValueError:
				continue

			tradi = chinese_tokenizer(tradi)
			verna =	chinese_tokenizer(verna)

			vocab.add_word_lst(tradi)
			vocab.add_word_lst(verna)

			_data.append(Instance(traditional = tradi , vernacular = verna))


	random.shuffle(_data)
	for x in _data:
		data.append(x)

	data.set_input("vernacular")
	data.set_target("traditional")
	return data


def indexize(data):
	data.apply(lambda x : [vocab.to_index(w) for w in x["vernacular"]] , new_field_name = "vernacular")
	data.apply(lambda x : [vocab.to_index(w) for w in x["traditional"]] , new_field_name = "traditional")
	
	return data

def run(path , force_reprocess , name = "data.pkl"):

	if (not force_reprocess) and os.path.exists(name):
		with open(name , "rb") as fil:
			ret = pickle.load(fil)
	else:
		train_data = load(path)
		train_data  = indexize(train_data)

		ret = vocab , train_data

	#pdb.set_trace()

	logger.log ("vocab len:"	, len(ret[0]))
	logger.log (" data len:" 	, len(ret[1]))

	with open("data.pkl" , "wb") as fil:
		pickle.dump(ret , fil)
	return ret

if __name__ == "__main__":
	run(C.data_path , C.force_reprocess)

	fitlog.add_best_metric(2333,"test")
	fitlog.finish()
