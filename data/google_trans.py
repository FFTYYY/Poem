from googletrans import Translator

t = Translator()

def translate_en_zh(s):
	return t.translate(s  , dest = "zh-CN").text

def convert(s):
	print (t.translate(s).text)
	return t.translate(t.translate(s).text , dest = "zh-CN").text

if __name__ == "__main__":
	#play
	while True:
		print (convert(input(">>")))
