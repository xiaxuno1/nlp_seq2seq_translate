#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: tokenizer.py
@date: 2025/10/25 16:25
@desc: 分词，中文按照子分词，英文使用NLTK提供的英文分词方法
"""
from tqdm import tqdm
from nltk.tokenize import TreebankWordTokenizer,TreebankWordDetokenizer


class BaseTokenizer:
    unk_token = '<unk>'
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self,vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}

        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @classmethod
    def tokenize(cls, text)->list[str]: #中文英文分成方法不同
        pass

    def encode(self, text,add_sos_eos=False):
        #根据seq2seq结构，在英文分词时需要加<sos><eos>
        tokens = self.tokenize(text)

        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token] #给text添加sos eos

        return [self.word2index.get(token,self.unk_token_index) for token in tokens] #返回索引

    @classmethod
    def build_vocab(cls,sentences,vocab_path):
        vocab = set()
        for sentence in tqdm(sentences,desc="building vocab"):
            vocab.update(cls.tokenize(sentence))

        vocab_list = ([cls.pad_token,cls.unk_token,cls.sos_token,cls.eos_token] +
                      [token for token in vocab if token.strip() != ''])
        print(f'vocab size: {len(vocab_list)}')

        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)

class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text)->list[str]:
        return list(text) #按字分词

class EnglishTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer() #添加类属性，方便类直接调用
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, text)->list[str]:
        return cls.tokenizer.tokenize(text) #调用接口实现分词
        
    def decode(self, indexes):
        """
        实现将硬卧的indexes转为word,nltk配套使用，可以实现一些特别的入：it's，优于普通
        :param text:
        :return:
        """
        tokens = [self.index2word.get(index) for index in indexes]
        return self.detokenizer.detokenize(tokens) #返回完整的英文句子

