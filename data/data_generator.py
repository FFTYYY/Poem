from data_utils import convert
import pdb
from tqdm import tqdm

raw_file = "./TangShi.txt"
save_file = "./CTangshi.txt"

def cut(s):
	#截短一个古诗

	return s[:96] # 96=6*8*2 五言诗保留16句，七言诗保留12句

with open(raw_file , "r" , encoding = "utf-8") as fil , open(save_file , "w" , encoding = "utf-8") as wfil:
	for line in tqdm(fil , desc = "Data Generating..."):

		#防止空串
		line = line.strip()
		if line == "":
			continue

		line = cut(line)

		#转换白话文
		trans = convert(line)

		#防止有多余分隔符
		line = line.replace("\t" , "")
		trans = trans.replace("\t" , "")

		#写入
		wfil.write(line + "\t" + trans + "\n")
		wfil.flush()

