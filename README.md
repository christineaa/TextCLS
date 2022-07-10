# TextCLS
TextCLS a toolkit and API for BERT-based text classification.
## Requirements and Installation
* python version=3.8
* ray-serve
```bash
pip install -r requirements.txt
pip install "ray[serve]" # for API
```

## Basic Usage
### Running with script
#### train
```bash
python nlptrainer.py --function bert_train --config_file path/your/config
```
#### evaluation
```bash
python nlptrainer.py --function bert_eval --config_file path/your/config
```
#### prediction
```bash
python nlptrainer.py --function bert_predict --config_file path/your/config
```
- An example of training configuration files is `./config/args.json`
- An example of evaluation/prediction configuration files is `./config/eval_args.json`
### Running with API
#### Server
```bash
python serve.py
```
#### Client
```python
import requests, json
# start training：
requests.post("http://127.0.0.1:9000/NLPServer/train", data=json.dumps({'config_path': 'config/args.json', 'task_id': 'job_1', "user_dir": "./"}))
# stop training
requests.post("http://127.0.0.1:9000/NLPServer/stop_train", data=json.dumps({'config_path': '', 'task_id': 'job_1', "user_dir": "./"}))
# start tensorboard
requests.post("http://127.0.0.1:9000/NLPServer/tensorboard", data=json.dumps({'port': '6006', "user_dir": "./"}))
# stop tensorboard
requests.post("http://127.0.0.1:9000/NLPServer/stop_tensorboard", data=json.dumps({'port': '6006', "user_dir": "./"}))
```

## Supported Functions

- change layer normalization position: post, pre
- change classifier head: fc, cnn, attention
- change activation functions: gelu, relu, squared_relu
- change pooling method: use cls, cls_before_pooler, avg, avg_top2, avg_first_last
- freeze parameters: all, encoder, embeddings, or some specific layers, e.g. 1,2,3
