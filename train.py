from config import C
from dataloader import run as load_data
from utils import logger  , fitlog_add_loss , fitlog_loss_step
import fitlog
from fastNLP import DataSetIter
from tqdm import tqdm
import pdb
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import pickle
from model import models
from transformers.optimization import get_cosine_schedule_with_warmup , get_linear_schedule_with_warmup
import time
import os

fitlog.commit(__file__)

def run(model , dataiter , loss_func , epoch_n = 0 , optim = None , scheduler = None , training = True):

	if training:
		model = model.train()
		run_name = "Training"
	else:
		model = model.eval()
		run_name = "Testing"

	pbar = tqdm(dataiter , ncols = 70)
	tot_loss = 0
	tot_step = 0
	for inp , tar in pbar:

		x = inp["vernacular"].cuda(C.gpus[0])
		y = tar["traditional"].cuda(C.gpus[0])

		y_inpt 	= y[:,:-1]
		y_gold 	= y[:,1:].contiguous()

		y_out = model(x , y_inpt)

		loss  = loss_func(y_out.view(-1,len(vocab)) , y_gold.view(-1))

		if training:
			optim.zero_grad()
			loss.backward()
			optim.step()
			scheduler.step()

		tot_loss += float(loss)
		tot_step += 1
		pbar.set_description_str( run_name + " on Epoch %d" % epoch_n)
		pbar.set_postfix_str    ( "avg loss: %.4f" % (tot_loss / tot_step))

		if training:
			fitlog_add_loss(float(loss)  		, epoch = epoch_n , name = "train loss")
			fitlog_add_loss(tot_loss / tot_step	, epoch = epoch_n , name = "avg train loss")


	return tot_loss / tot_step

def train(model , train_data , test_data):
	train_iter = DataSetIter(train_data , batch_size = C.batch_size)
	test_iter  = DataSetIter(test_data  , batch_size = C.batch_size)

	loss_func = nn.CrossEntropyLoss(ignore_index = 0)
	optim = tc.optim.Adam(params = model.parameters() , lr = C.lr , weight_decay = C.weight_decay)	
	scheduler = get_cosine_schedule_with_warmup(
		optim , 
		num_warmup_steps = C.warmup ,
		num_training_steps = train_iter.num_batches * C.epoch_number , 
	)

	best_test_loss 	= -1
	best_test_epoch = -1
	best_step 		= -1
	try:
		for epoch_n in range(C.epoch_number):
			tra_loss = run(model , train_iter , loss_func , epoch_n , optim , scheduler , True)
			tes_loss = run(model , test_iter , loss_func , epoch_n , None , None , False)

			logger.log ("Epoch %d ended. Train loss = %.4f , Valid loss = %.4f" % (
				epoch_n , tra_loss , tes_loss ,
			))
			fitlog.add_metric(
				tes_loss , 
				step = train_iter.num_batches * (epoch_n + 1) , 
				epoch = epoch_n , 
				name = "valid loss"
			)

			if best_test_epoch < 0 or tes_loss < best_test_loss:
				best_test_loss = tes_loss
				best_test_epoch = epoch_n
				best_step = fitlog_loss_step["train loss"]

				fitlog.add_best_metric(best_test_loss , name = "loss")
				with open(C.model_save , "wb") as fil:#暂时保存目前最好的模型
					pickle.dump(model , fil)
				fitlog.add_hyper(name = "best_step" , value =  "%d / %d" % (
					best_step ,
					train_iter.num_batches * C.epoch_number , 
				))

	except KeyboardInterrupt: # 手动提前停止
		pass

	logger.log ("Train end.")
	logger.log ("Got best valid loss %.4f in epoch %d" % (best_test_loss , best_test_epoch))

	return model

if __name__ == "__main__":

	#----- get data & model -----

	vocab , data = load_data(C.data_path , C.force_reprocess , C.data_save)
	train_data , valid_data = data[:-1000] , data[-1000:]

	Model = models[C.model]
	model = Model(
		vocab = vocab , logger = logger ,
		d_model = C.d_model , num_layers = C.num_layers , d_hid = C.d_hid , h = C.h ,
		dropout = C.dropout ,
	 )
	model = model.cuda(C.gpus[0])
	if len(C.gpus) > 1:
		#tc.distributed.init_process_group(backend = "nccl")
		model = nn.DataParallel(model , C.gpus)

	#----- train -----

	start_time = time.time()
	model = train(model , train_data , valid_data)
	end_time = time.time()
	fitlog.add_hyper(name = "training time" , value = "%.3f" % (end_time - start_time))


	#----- save model -----
	logger.log("model saved.")

	#----- finish -----

	fitlog.finish()