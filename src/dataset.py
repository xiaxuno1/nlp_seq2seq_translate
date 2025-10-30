#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: dataset.py
@date: 2025/10/27 14:08
@desc:数据集
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from src import config


class TranslationDataset(Dataset):
    def __init__(self,path):
        self.data = pd.read_json(path, lines=True,orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.data[idx]['zh'],dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['en'],dtype=torch.long)
        return input_tensor, target_tensor

def collate_fn(batch):
    """
    把不同长度的序列批量化，collata_fn决定了如何把一个batch的样本从列表 组合成张量
    :param batch:是一个列表，里面每个元素来自 __getitem__()
    如：[(input_tensor1, target_tensor1), (input_tensor2, target_tensor2), ...]
    :return:
    """
    input_tensor, target_tensor = zip(*batch) #解包 为tuple（input）,tuple（target）
    input_tensor = pad_sequence(input_tensor, batch_first=True,padding_value=0) #对队列进行补齐
    target_tensor = pad_sequence(target_tensor, batch_first=True,padding_value=0) #根据tuple内的最大长度进行补齐
    return input_tensor, target_tensor

def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ('train_dataset.jsonl' if train else 'test_dataset.jsonl')
    dataset = TranslationDataset(path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE,shuffle=True,collate_fn=collate_fn) # 每个句子的长度不一，需要collata_fn处理


if __name__ == '__main__':
    dataloader = get_dataloader()
    for input_tensor,target_tensor in dataloader:
        print(input_tensor.shape)
        print(input_tensor[0])
        print(target_tensor[0])
        print(target_tensor.shape)
        print(input_tensor[0])
        print(target_tensor[0])
        break