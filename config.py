import fitlog
#fitlog.debug()
import argparse
from model import models
from utils import autoname
import os
import time

fitlog.set_log_dir("logs")
fitlog.commit(__file__)

_par = argparse.ArgumentParser()

#---------------------------------------------------------------------------------------------------

#----- data process -----
_par.add_argument("--data_path"		, type = str 	, default = "./data/Verna_Tangshi.txt")
_par.add_argument("--force_reprocess"  , action = "store_true" , default = False)

#----- model cofig -----
_par.add_argument("--model" 		, type = str  	, default = "transformer" , choices = list(models))
_par.add_argument("--d_model"		, type = int 	, default = 512)
_par.add_argument("--dropout"		, type = float 	, default = 0.0)

_par.add_argument("--num_layers"	, type = int 	, default = 4)
_par.add_argument("--h"				, type = int 	, default = 8)
_par.add_argument("--d_hid"			, type = int 	, default = 2048)

#----- training procedure -----
_par.add_argument("--batch_size"	, type = int 	, default = 128)
_par.add_argument("--epoch_number"	, type = int 	, default = 10)
_par.add_argument("--lr"			, type = float 	, default = 1e-4)
_par.add_argument("--weight_decay"	, type = float 	, default = 1e-8)
_par.add_argument("--warmup" 		, type = int 	, default = 400)

#----- others -----
_par.add_argument("--seed" 			, type = int 	, default = 2333)
_par.add_argument("--model_save" 	, type = str 	, default = "model.pkl")
_par.add_argument("--data_save" 	, type = str 	, default = "data.pkl")
_par.add_argument("--gpus" 			, type = str 	, default = "0")
_par.add_argument("--name"   		, type = str 	, default = "")

_par.add_argument("--gene_input" 	, type = str 	, default = "")


#---------------------------------------------------------------------------------------------------

C = _par.parse_args()

if not C.name:
	C.name = autoname()
os.makedirs("./model_save" , exist_ok = True)
C.model_save = "./model_save/model_%s" % C.name

now_time = time.localtime(time.time())
C.time = "%d-%d-%d %d:%d" % ( 
	(now_time.tm_year) % 100 ,  
	now_time.tm_mon ,  
	now_time.tm_mday ,  
	now_time.tm_hour , 
	now_time.tm_min , 
)

fitlog.add_hyper(C)


def listize(s):
	return [int(x) for x in s.strip().split(",")]
C.gpus = listize(C.gpus)


if C.seed >= 0:
	fitlog.set_rng_seed(C.seed)
else:
	fitlog.set_rng_seed()
