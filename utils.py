import time
import fitlog
import pdb

fitlog.commit(__file__)

class logger:
	def log(*args):
		x = " ".join([str(x) for x in args])
		print(x)
		fitlog.add_to_line(x)


fitlog_loss_step = {}
def fitlog_add_loss(value , epoch , name):
	if fitlog_loss_step.get(name) is None:
		fitlog_loss_step[name] = 0
	fitlog_loss_step[name] += 1

	fitlog.add_loss(value = value , step = fitlog_loss_step[name] , epoch = epoch , name = name)


def autoname():
	dic = {"seed" : time.time()}
	def next():
		dic["seed"] = (dic["seed"] * dic["seed"] + 3 * dic["seed"] + 2333) % (1e9+7)
		return int(dic["seed"])
	a = "plmnbtrfdwsky"
	b = "aeiou"

	name = ""
	for u in range(next() % 6 + 3):
		now_s = [a,b][u % 2]
		name += now_s[next() % (len(now_s))]
	return name
