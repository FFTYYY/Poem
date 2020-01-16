Poem：从白话文生成古诗。

## 训练和生成
1. 训练：`python train.py --gpus=0,1,2,3 --name=<name>`
2. 生成：`python generate.py --name=<name>`

我是在4块gpu上训练的，如果gpu不够或者显存太小可以适当把batch_size改小（通过`--batch_size`指令），不过这么干有可能无法复现结果。

## 数据集复用
1. 本文使用的数据集Verna_Tangshi文件保存在data/Verna_Tangshi.txt
2. 可以调用data/data_generator.py从现有的诗集生成白话-诗歌数据集。创建一个新文件data/Tangshi.txt，把作为数据的诗歌写进去（每首一行），然后调用`python data_generator.py`就可以了
3. 注意，调用data_generator.py之前需要先注册百度翻译api账号，注册之后创建一个寻文件data/baidu_ids.py，将你的id写进去，格式类似于：
```
ids = {
	"id1" : [
		'xxxxx' , # id
		'xxxxx' , # 密码
	 ] ,
}
```
可以使用多个id轮流调用以加快生成速度，或者使用Google翻译，需要自行改代码。
