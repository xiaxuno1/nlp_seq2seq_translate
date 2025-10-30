#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: model.py
@date: 2025/10/27 10:18
@desc: seq2seq模型
"""
import torch
from torch import nn

from src import config


class TranslationEncoder(nn.Module):
    """
    定义编码器
    """
    def __init__(self,vocab_size,padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM, padding_idx=padding_index)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)

    def forward(self,x):
        # x:[batch_size,seq_len]
        embedded = self.embedding(x)
        #embedded :[batch_seize,seq_len,embed_dim]
        output, hidden = self.gru(embedded)
        # output:[batch_size,seq_len,hidden_size]
        # hidden:[1，batch_size,hidden_size]

        # 提取最后一个有效token的hidden作为decoder的输入
        lengths = (x != self.embedding.padding_idx).sum(dim =1) #提起每个bathc_size的有效长度，降维 [batch_size]
        last_hidden_state = output[torch.arange(output.shape[0]), lengths-1] #output[[0,1,2,..bathc_size],[len,len]]

        return last_hidden_state # [batch_size,hidden_size] 取出每个样本的最后一个有效隐藏状态

class TranslationDecoder(nn.Module):
    """
    定义解码器
    """
    def __init__(self,vocab_size,padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)
        self.out = nn.Linear(in_features=config.HIDDEN_SIZE,
                             out_features=vocab_size) # 通过线性层输出下一个token的预测

    def forward(self,x,hidden_0):
        """

        :param x: 输入，
        :param hidden_0:接收encoder的hidden_state作为h_0
        :return:
        """
        #x:[batch_size,1]  每一个输入token都要输出一个预测，计算loss;所有loss求和得到seq_len的loss
        #hidden_0:[1,batch_size,hidden_size]
        embed = self.embedding(x)
        # embed:[batch_size,1,em_dim]
        output, hidden_n = self.gru(embed, hidden_0) #指定hidden_size为encoder的last_output_state
        #output:[batch_size,1,hidden_size]
        #hidden_n:[1,batch_size,hidden_size]
        output = self.out(output)
        # [batch_size,1,vocab_size] 每个输出一个vocab_size大小的概率分布，用于预测
        return output, hidden_n


class TranslationSeq2Seq(nn.Module):
    def __init__(self,zh_vocab_size,en_vocab_size,zh_padding_index,en_padding_index):
        super().__init__()
        self.encoder = TranslationEncoder(vocab_size=zh_vocab_size,padding_index=zh_padding_index)
        self.decoder = TranslationDecoder(vocab_size=en_vocab_size,padding_index=en_padding_index)