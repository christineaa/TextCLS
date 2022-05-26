from torch.utils.data import Dataset
import pandas as pd
# from tqdm import tqdm
import torch

# tqdm.pandas()

def clean_symbols(content):
    return content


class BertDataset(Dataset):
    """数据集的创建"""

    def __init__(self, path, tokenizer, config):
        super(BertDataset, self).__init__()
        self.data = pd.read_csv(path, encoding='utf_8_sig')
        content_name = config.src_column1
        label_name = config.tgt_column
        self.data["sentence"] = self.data[content_name]
        self.data["cut_sentence"] = self.data['sentence'].apply(clean_symbols)
        # 标签映射到id
        label2id = {label: i for i, label in enumerate(self.data[label_name].unique())}
        self.data['category_id'] = self.data[label_name].apply(lambda x: x.strip()).map(label2id)
        self.tokenizer = tokenizer
        self.max_seq_length = config.max_seq_length
        self.label_name = label_name
        self.num_labels = len(label2id)
        self.label2id = label2id
        self.pad_to_max_length = config.pad_to_max_length

    def __getitem__(self, i):
        data = self.data.iloc[i]
        text = data['cut_sentence']  # text数据
        text_dict = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=True
        )
        input_ids, attention_mask, token_type_ids = text_dict['input_ids'], \
                                                    text_dict['attention_mask'], \
                                                    text_dict['token_type_ids']

        output = {
            "input_ids": input_ids,
            'attention_mask': attention_mask,
            "token_type_ids": token_type_ids,
        }
        if self.label_name:
            labels = int(data['category_id'])
            output["labels"] = labels
        return output

    def __len__(self):
        return self.data.shape[0]


def collate_fn(batch):
    """
    动态padding,返回Tensor
    :param batch:
    :return: 每个batch id和label
    """
    def padding(indice, max_length, pad_idx=0):
        """
        填充每个batch的句子长度
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = len(token_ids[0])  # batch中样本的最大的长度
    token_type_ids = [data["token_type_ids"] for data in batch]
    attention_mask = [data["attention_mask"] for data in batch]
    # 填充每个batch的sample
    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    attention_mask_padded = padding(attention_mask, max_length)
    if 'labels' in batch[0]:
        labels = torch.tensor([data["labels"] for data in batch])
        return token_ids_padded, attention_mask_padded, token_type_ids_padded, labels
    return token_ids_padded, attention_mask_padded, token_type_ids_padded
