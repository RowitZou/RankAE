# RankAE

Pytorch implementation of the AAAI-2021 paper: [Unsupervised Summarization for Chat Logs with Topic-Oriented Ranking and Context-Aware Auto-Encoders](https://arxiv.org/pdf/2012.07300).

The code is partially referred to https://github.com/nlpyang/PreSumm.

## Requirements

* Python 3.6 or higher
* torch==1.1.0
* pytorch-transformers==1.1.0
* torchtext==0.4.0
* rouge==0.3.2
* tensorboardX==2.1
* nltk==3.5

## Environment

* Tesla V100 32GB GPU
* CUDA 10.2

## Data Format

Each json file is a data list that includes chat log samples. The format of a chat log sample is shown as follows:

```
{"session": [
    {"content": ["is", "anyone", "there", ",", "please", "?"],
	 "type": "c2b"},
    {"content": ["i", "want", "to", "buy", "this", "skirt", ",", "but", "i", "don't", "know", "what", "size", "suits", "me"],
	 "type": "c2b"}, 
    {"content": ["what's", "your", "height", "and", "weight", "?"],
	 "type": "b2c"}, 
    {"content": ["165", "cm", "and", "55", "kg"],
	 "type": "c2b"}, 
    {"content": ["well", ",", "size", "m", "suits", "you"],
	 "type": "b2c"}
 ],
 "summary": ["the", "user", "wants", "to", "buy", "a", "skirt", ".", "size", "m", "suits", "people", "of", "165", "cm", "and", "55", "kg", "."]
}
```

```
{"session": [
    {"content": ["发", "什", "么", "快", "递", "?"],
	 "type": "c2b"},
    {"content": ["发", "顺", "丰"],
	 "type": "b2c"}, 
    {"content": ["包", "邮", "吗"],
	 "type": "c2b"}, 
    {"content": ["满", "300", "元", "包", "邮"],
	 "type": "b2c"}, 
    {"content": ["我", "下", "单", "了", "，", "什", "么", "时", "候", "发", "货"],
	 "type": "c2b"},
    {"content": ["明", "天"],
	 "type": "b2c"}
 ],
 "summary": ["商", "品", "明", "天", "发", "顺", "丰", "，", "满", "300", "元", "包", "邮", "。"]
}
```

## Usage

* Download BERT checkpoints.

	The pretrained BERT checkpoints can be found at:
	
	* Chinese BERT: https://github.com/ymcui/Chinese-BERT-wwm
	* English BERT: https://github.com/google-research/bert

	Put BERT checkpoints into the directory **bert** like this:

	```
	--- bert
	  |
	  |--- chinese_bert
	     |
		 |--- config.json
		 |
		 |--- pytorch_model.bin
		 |
		 |--- vocab.txt
	```

* Data Processing

	```
	PYTHONPATH=. python ./src/preprocess.py -raw_path json_data -save_path bert_data -bert_temp_dir bert/chinese_bert -log_file logs/preprocess.log
	```
* Train

	```
	PYTHONPATH=. python ./src/train.py -data_path bert_data/taobao -log_file logs/rankae.train.log -model_path models/rankae -sep_optim -train_steps 200000
	```

* Validate

	```
	PYTHONPATH=. python ./src/train.py -mode validate -data_path bert_data/taobao -log_file logs/rankae.val.log -alpha 0.95 -model_path models/rankae
	```

* Testing

	```
	PYTHONPATH=. python ./src/train.py -mode test -data_path bert_data/taobao -test_from models/rankae/model_step_200000.pt -log_file logs/rankae.test.log -alpha 0.95
	```
		               
## Data

Our chat log dataset is collected from [Taobao](https://www.taobao.com/), where conversations take place between customers and merchants in the Chinese language. For the security of private information from customers, we performed the data desensitization and converted words to IDs.

The desensitized data is available at 
[Google Drive](https://drive.google.com/file/d/1DZalpN2uKer9oiR8xjaL2nj3p1jFaGGj/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1570uHnC-bxs2kYYRoWG7SA) (extract code: 4298).
