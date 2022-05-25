import ray
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel

from nlptrainer import bert_train, bert_predict


class Item(BaseModel):
    config_path: str


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
        bert_train(item.config_path)
        # print(item.config_path)
        return "success"

    @app.post("/predict")
    def setup(self, config_path: str):
        bert_predict(config_path)
        return "success"


NLPServer.deploy()

# 调用：requests.post("http://127.0.0.1:9000/NLPServer/train", data=json.dumps({'config_path': 'args.json'}))