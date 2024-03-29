import os
import requests
import json
import time
import ray
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel

from nlptrainer import bert_train, bert_predict


class Item(BaseModel):
    config_path: str
    user_dir: str
    task_id: str


class TensorboardItem(BaseModel):
    port: str
    user_dir: str


os.environ["RAY_LOG_TO_STDERR"] = "1"
ray.init(address="auto")
serve.start(http_options=dict(
    host="127.0.0.1",
    port=9000,

    # root_path="/root"
))

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class NLPServer:
    def __init__(self):
        self.count = 0

    @app.get("/")
    def get(self):
        return "success"

    @app.post("/train")
    def setup(self, item: Item):
        print("train api", item.config_path, item.user_dir, item.task_id)
        os.makedirs(item.user_dir, exist_ok=True)
        cmd = f"python nlptrainer.py --config_file={item.config_path} --function=bert_train --task_id={item.task_id}" \
              f"& echo $! > {item.user_dir}/train.pid"
        os.system(cmd)
        # bert_train(item.config_path)
        return "success"

    @app.post("/stop_train")
    def setup(self, item: Item):
        cmd = f"kill -9 `cat {item.user_dir}/train.pid`"
        res = os.system(cmd)
        return "success" if res == 0 else "fail"

    @app.post("/predict")
    def setup(self, item: Item):
        print("predict api", item.config_path, item.user_dir, item.task_id)
        os.makedirs(item.user_dir, exist_ok=True)
        cmd = f"python nlptrainer.py --config_file={item.config_path} --function=bert_predict --task_id={item.task_id}" \
              f"& echo $! > {item.user_dir}/predict.pid"
        os.system(cmd)
        # bert_predict_interface(item.config_path, is_infer=True)
        return "success"

    @app.post("/stop_predict")
    def setup(self, item: Item):
        cmd = f"kill -9 `cat {item.user_dir}/predict.pid`"
        res = os.system(cmd)
        return "success" if res == 0 else "fail"

    @app.post("/evaluate")
    def setup(self, item: Item):
        print("evaluate api", item.config_path, item.user_dir, item.task_id)
        os.makedirs(item.user_dir, exist_ok=True)
        cmd = f"nohup python nlptrainer.py --config_file={item.config_path} --function=bert_eval --task_id={item.task_id}" \
              f"& echo $! > {item.user_dir}/evaluation.pid"
        os.system(cmd)
        # bert_predict_interface(item.config_path, is_infer=False)
        return "success"

    @app.post("/stop_evaluation")
    def setup(self, item: Item):
        cmd = f"kill -9 `cat {item.user_dir}/evaluation.pid`"
        res = os.system(cmd)
        return "success" if res == 0 else "fail"

    @app.post("/tensorboard")
    def setup(self, item: TensorboardItem):
        dir = os.path.join(item.user_dir, "tensorboard")
        if not os.path.exists(dir):
            dir = "/root/TextCLS/output/tensorboard"
        cmd = f"nohup tensorboard --logdir={dir} --port={item.port} --host=0.0.0.0 " \
              f"> {item.user_dir}/tensorboard.out 2>&1 & echo $! > {item.user_dir}/tensorboard.pid"
        os.system(cmd)
        time.sleep(600)
        cmd = f"kill -9 `cat {item.user_dir}/tensorboard.pid`"
        res = os.system(cmd)
        return "success" if res == 0 else "fail"

    @app.post("/stop_tensorboard")
    def setup(self, item: TensorboardItem):
        cmd = f"kill -9 `cat {item.user_dir}/tensorboard.pid`"
        res = os.system(cmd)
        return "success" if res == 0 else "fail"

NLPServer.deploy()

while True:
    time.sleep(5)
    # print(serve.list_deployments())
# 调用train：requests.post("http://127.0.0.1:9000/NLPServer/train", data=json.dumps({'config_path': 'config/args.json', 'task_id': '22072667663971', "user_dir": "./"}))
# 停止train：requests.post("http://127.0.0.1:9000/NLPServer/stop_train", data=json.dumps({'config_path': '', 'task_id': 'job_1', "user_dir": "./output"}))
# 启用tensorboard：requests.post("http://127.0.0.1:9000/NLPServer/tensorboard", data=json.dumps({'port': '6006', "user_dir": "./output"}))