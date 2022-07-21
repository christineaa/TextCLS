import logging
import os
import time
import requests
import json
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import numpy as np
import argparse
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
)
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from transformers.trainer_utils import is_main_process
from transformers.integrations import TensorBoardCallback
from transformers import EarlyStoppingCallback

from utils.dataset import BertDataset
from model.bert import OurBertForSequenceClassification
from model.modified_bert import CustomizedBertConfig


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--function", type=str, required=True, choices=["bert_train", "bert_predict", "bert_eval"])
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--task_id", type=str, required=True)
    args = parser.parse_args()
    return args


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # customized arguments
    ln_type: str = field(
        default="post",
        metadata={
            "help": "What kind of layer normalization to use (post-ln, pre-ln)."
        }
    )
    freeze: str = field(
        default="",
        metadata={
            "help": "Which layers should be frozen.(all, encoder, embeddings)"
        }
    )
    freeze_layers: str = field(
        default="",
        metadata={
            "help": "Which layers should be frozen."
        }
    )
    cls_type: str = field(
        default="FC",
        metadata={
            "help": "What kind of classifier to use after backbone(FC, lstm, testcnn)."
        }
    )
    activation: str = field(
        default="gelu",
        metadata={
            "help": "What kind of activation function to use in BERT(gelu, relu, squared_relu)."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last), only used "
                    "when cls_type=FC. "
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments.
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # customized arguments
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training data file (.csv)."}
    )
    valid_file: Optional[str] = field(
        default=None,
        metadata={"help": "The validation data file (.csv)."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "The testing data file (.csv)."}
    )
    src_column1: Optional[str] = field(
        default=None,
        metadata={"help": "which column to be input1"}
    )
    src_column2: Optional[str] = field(
        default=None,
        metadata={"help": "which column to be input2"}
    )
    tgt_column: Optional[str] = field(
        default=None,
        metadata={"help": "which column to be label"}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.valid_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "tsv"], "`train_file` should be a csv, or a tsv file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


def bert_train(config_path):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=config_path)
    # set log
    logger = logging.getLogger(__name__)

    os.makedirs(training_args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(training_args.output_dir, "log"), mode="w")
    logging.basicConfig(
        format='[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] >> %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[file_handler],
        level=logging.INFO
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        # transformers.utils.logging.disable_default_handler()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        transformers.utils.logging.add_handler(file_handler)

    logger.info("Training/evaluation parameters %s", training_args)

    config = CustomizedBertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        ln_type=model_args.ln_type,
        pooler_type=model_args.pooler_type,
        cls_type=model_args.cls_type,
        freeze=model_args.freeze,
        freeze_layers=model_args.freeze_layers,
        model_name_or_path=model_args.model_name_or_path,
        hidden_act=model_args.activation
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    train_dataset = BertDataset(data_args.train_file, tokenizer, data_args)
    dev_dataset = BertDataset(data_args.valid_file, tokenizer, data_args)
    config.label2id = train_dataset.label2id
    config.num_labels = train_dataset.num_labels
    config.id2label = {v: k for k, v in train_dataset.label2id.items()}

    if "roberta" in model_args.model_name_or_path or \
        "albert" in model_args.model_name_or_path or \
        "distilbert" in model_args.model_name_or_path:
        model = OurBertForSequenceClassification(config=config)
    else:
        model = OurBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            cache_dir=model_args.cache_dir
        )

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if training_args.fp16 else None))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=dev_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[TensorBoardCallback, EarlyStoppingCallback(early_stopping_patience=5)],
        compute_metrics=compute_metrics
    )
    # trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    train_result = trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "latest"))
    output_train_file = os.path.join(training_args.output_dir, "train_results.json")

    if trainer.is_world_process_zero():
        # tokenizer.save_pretrained(training_args.output_dir)
        logger.info("***** Train results *****")
        json.dump(train_result[2], open(output_train_file, "w"), indent=2)
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f"  {key} = {value}")

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_dataset=dev_dataset)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.json")
        if trainer.is_world_process_zero():
            logger.info("***** Eval results *****")
            json.dump(results, open(output_eval_file, "w"), indent=2)
            for key, value in sorted(results.items()):
                logger.info(f"  {key} = {value}")
    return


def bert_predict(config_path):
    return bert_predict_interface(config_path, is_infer=True)


def bert_eval(config_path):
    return bert_predict_interface(config_path, is_infer=False)


def bert_predict_interface(config_path, is_infer=True):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=config_path)
    # set log
    logger = logging.getLogger(__name__)

    os.makedirs(training_args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(training_args.output_dir, "log"), mode="w")
    logging.basicConfig(
        format='[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] >> %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[file_handler],
        level=logging.INFO
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed testing: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.disable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        transformers.utils.logging.add_handler(file_handler)

    logger.info("Training/evaluation parameters %s", training_args)

    config = CustomizedBertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        hidden_act=model_args.activation,
        ln_type=model_args.ln_type,
        pooler_type=model_args.pooler_type,
        cls_type=model_args.cls_type,
        freeze=model_args.freeze,
        freeze_layers=model_args.freeze_layers
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    test_dataset = BertDataset(data_args.test_file, tokenizer, data_args)

    model = OurBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )
    model.eval()
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if training_args.fp16 else None))

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    eval_out = evaluate(model, test_dataloader, training_args.device, is_infer)
    if is_infer:
        pred_logits, pred_idx = eval_out
        logits_file = os.path.join(training_args.output_dir, "predict_probs.json")
        pred_logits = pred_logits.tolist()
        logits = {i: logit for i, logit in enumerate(pred_logits)}
        json.dump(logits, open(logits_file, "w"), indent=2)

        pred_label = [config.id2label[idx] for idx in pred_idx]
        pred_df = pd.read_csv(data_args.test_file, encoding='utf_8_sig')
        pred_df["prediction"] = pred_label
        output_file = os.path.join(training_args.output_dir, "predict.csv")
        pred_df.to_csv(output_file)

    else:
        test_acc, micro_result, macro_result, test_loss, test_report, test_confusion = eval_out
        results = {
            "label": config.id2label,
            "accuracy": test_acc,
            "loss": test_loss.item(),
            "micro_precision": micro_result[0],
            "micro_recall": micro_result[1],
            "micro_F1": micro_result[2],
            "macro_precision": macro_result[0],
            "macro_recall": macro_result[1],
            "macro_F1": macro_result[2],
            "report": test_report,
            "confusion_matrix": test_confusion.tolist()
        }
        sn.heatmap(test_confusion, annot=True, cmap="YlGnBu",
                   xticklabels=config.id2label.values(), yticklabels=config.id2label.values())
        plt.savefig(os.path.join(training_args.output_dir, 'heatmap.png'), dpi=300)
        output_file = os.path.join(training_args.output_dir, "eval_results.json")
        json.dump(results, open(output_file, "w"), indent=2)

        msg = 'Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}'
        logger.info(msg.format(test_loss, test_acc))
        msg = "Micro Precision: {0:>6.2%}, Micro Recall: {1:>6.2%}, Micro F1: {2:>6.2%}"
        logger.info(msg.format(*micro_result[:-1]))
        msg = "Macro Precision: {0:>6.2%}, Macro Recall: {1:>6.2%}, Macro F1: {2:>6.2%}"
        logger.info(msg.format(*macro_result[:-1]))
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(config.id2label)
        logger.info(test_confusion)
    return


def evaluate(model, data_iter, device, is_infer=False):
    loss_total = 0
    predict_all = []
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
            if is_infer:
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                labels = batch["labels"].to(device, non_blocking=True)
                outputs = model(input_ids, attention_mask, token_type_ids, labels=labels)
                loss = outputs.loss
                loss_total += loss
                labels = labels.data.cpu().numpy()
                labels_all = np.append(labels_all, labels)

            pred_probs = torch.softmax(outputs.logits, dim=-1)
            predict_all.extend(pred_probs.cpu().numpy())

    predict_all = np.array(predict_all)
    predict_idx = np.argmax(predict_all, axis=-1)
    if is_infer:
        return predict_all, predict_idx

    acc = accuracy_score(labels_all, predict_idx)
    micro_result = precision_recall_fscore_support(labels_all, predict_idx, average='micro')
    macro_result = precision_recall_fscore_support(labels_all, predict_idx, average='macro')
    report = classification_report(labels_all,
                                   predict_idx,
                                   target_names=model.config.id2label.values(),
                                   digits=4)
    confusion = confusion_matrix(labels_all, predict_idx)
    return acc, micro_result, macro_result, loss_total / len(data_iter), report, confusion


if __name__ == "__main__":
    args = setup_args()
    if os.getcwd() == '/root/TextCLS':
        time.sleep(600)
    else:
        eval(args.function)(args.config_file)
    requests.post("http://127.0.0.1:8088/train", data=json.dumps({'task_id': args.task_id, 'status': 'finish'}))
    # bert_train("config/args.json")
    # bert_predict("config/eval_args.json")