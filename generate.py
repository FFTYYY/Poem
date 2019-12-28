import fitlog
fitlog.debug()
from config import C
from model.generate import generate as model_generate
import torch as tc
from dataloader import chinese_tokenizer
import pdb
import pickle

with open(C.model_save , "rb") as fil:
	model = pickle.load(fil)
#model = tc.load(C.model_save)
if isinstance(model , tc.nn.DataParallel):
	model = model.module
model = model.eval().cuda(C.gpus[0])

def generate_from_sents(model , sent , return_index = False):

	sent = sent.replace(" " , "").replace("\t" , "").replace("\n" , "")
	sent = chinese_tokenizer(sent)
	sent = [model.vocab.to_index(w) for w in sent]

	x = tc.LongTensor(sent).cuda(C.gpus[0])
	y = model_generate(model , x)

	if return_index:
		return y

	y = [model.vocab.to_word(w) for w in list(y)]
	return "".join(y)



if __name__ == "__main__":
	while True:

		sent = input(">>")
		if sent == "q":
			break
		#sent = """晚霞，山川，云"""
		print ()
		print (generate_from_sents(model , sent))
		print ()