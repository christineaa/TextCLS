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
    task_id: str


ray.init()
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
        try:
            bert_train(item.config_path)
        except Exception as e:
            requests.post("http://127.0.0.1:8080/train", data=json.dumps({'task_id': item.task_id, 'status': repr(e)}))
        requests.post("http://127.0.0.1:8080/train", data=json.dumps({'task_id': item.task_id, 'status': 'finish'}))
        return "success"

    @app.post("/predict")
    def setup(self, item: Item):
        try:
            bert_predict(item.config_path, is_infer=True)
        except Exception as e:
            requests.post("http://127.0.0.1:8080/predict", data=json.dumps({'task_id': item.task_id, 'status': repr(e)}))
        requests.post("http://127.0.0.1:8080/predict", data=json.dumps({'task_id': item.task_id, 'status': 'finish'}))
        return "success"

    @app.post("/evaluate")
    def setup(self, item: Item):
        try:
            bert_predict(item.config_path, is_infer=False)
        except Exception as e:
            requests.post("http://127.0.0.1:8080/evaluate", data=json.dumps({'task_id': item.task_id, 'status': repr(e)}))
        requests.post("http://127.0.0.1:8080/evaluate", data=json.dumps({'task_id': item.task_id, 'status': 'finish'}))
        return "success"


NLPServer.deploy()

while True:
    time.sleep(5)
    # print(serve.list_deployments())
# 调用：requests.post("http://127.0.0.1:9000/NLPServer/train", data=json.dumps({'config_path': 'args.json', 'task_id': 'job_1'}))