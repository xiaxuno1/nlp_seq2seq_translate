#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: process.py
@date: 2025/10/23 15:20
@desc: 处理原始的jsonl数据
"""
from tqdm import tqdm
import pandas as pd
from src import config
from sklearn.model_selection import train_test_split

from src.tokenizer import ChineseTokenizer, EnglishTokenizer


def process():
    """

    :return:
    """
    # 读取无头的tsv,提取，命名
    df = pd.read_csv(config.RAW_DATA_DIR / 'cmn.txt', sep='\t', header=None, usecols=[0,1], names=['en','zh'],
                     encoding='utf-8').dropna()

    #划分数据集和测试集
    train_df, test_df = train_test_split(df, test_size = 0.2)
    print("数据集和测试集划分完成：",len(train_df),len(test_df))
    print(train_df['zh'].iloc[0],train_df['en'].iloc[0])
    print(test_df['zh'].iloc[0], test_df['en'].iloc[0]) #pands的dataFrame默认使用索引标签而不是行号，当打乱，索引就不在是0,1,2

    # 分别构建tokenizer
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(),config.MODELS_DIR / 'zh_vocab.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(),config.MODELS_DIR / 'en_vocab.txt')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / "zh_vocab.txt") #读取词表，返回的是cls，因此可以直接调用encode
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / "en_vocab.txt")

    # tokenizer encode
    train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x,add_sos_eos=False))
    train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x,add_sos_eos = True))
    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x,add_sos_eos=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x,add_sos_eos = True))
    print("编码完成encode:",len(train_df),len(test_df))
    print(train_df.shape,test_df.shape)
    print(train_df.head(1))

    #保存
    pd.DataFrame(train_df).to_json(config.PROCESSED_DATA_DIR / 'train_dataset.jsonl',lines=True,orient="records")
    pd.DataFrame(test_df).to_json(config.PROCESSED_DATA_DIR / "test_dataset.jsonl",lines=True,orient="records")

if __name__ == '__main__':
    process()