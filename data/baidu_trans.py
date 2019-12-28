import http.client
import hashlib
import json
import urllib
import random
import opencc
import pdb
import time
from baidu_ids import ids

def translate(content , id_use = "id1" , src_lan = "zh" , tar_lan = "en"):
	appid = ids[id_use][0]
	secretKey = ids[id_use][1]
	httpClient = None
	myurl = '/api/trans/vip/translate'
	q = content
	salt = random.randint(32768, 65536)
	sign = appid + q + str(salt) + secretKey
	sign = hashlib.md5(sign.encode()).hexdigest()
	myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
		q) + '&from=' + src_lan + '&to=' + tar_lan + '&salt=' + str(
		salt) + '&sign=' + sign
 
	try:
		httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
		httpClient.request('GET', myurl)

		response = httpClient.getresponse()
		jsonResponse = response.read().decode("utf-8")
		js = json.loads(jsonResponse)
		dst = str(js["trans_result"][0]["dst"])
		return dst
	except Exception:
		return None
	finally:
		if httpClient:
			httpClient.close()

def translate_zh_en(s):
	return translate(s , "id1" , "zh" , "en")
def translate_en_zh(s):
	return translate(s , "id1" , "en" , "zh")

if __name__ == '__main__':
	while True:
		s = input(">>")
		s = translate_zh_en(s)
		time.sleep(1.1)
		s = translate_en_zh(s)
		print (s)

