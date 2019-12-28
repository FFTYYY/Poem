from baidu_trans import translate_en_zh
from baidu_trans import translate_zh_en
import time

def convert(s):

	eng = translate_zh_en(s)

	while eng is None:
		time.sleep(1.1)
		eng = translate_zh_en(s)

	han = translate_en_zh(eng)

	while han is None:
		time.sleep(1.1)
		han = translate_en_zh(eng)
		
	return han

if __name__ == "__main__":
	#play
	while True:
		print (convert(input(">>")))
