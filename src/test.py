#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: test.py
@date: 2025/10/25 10:10
@desc: 
"""
import random

import torch

output = torch.rand(4,3,2) # [batch_size,seq_len,hidden_size]
print(output)
print(output[3,2]) #取出具体的索引的tensor
print(output.shape[1]) #提取轴1的长度
print(output.shape[-1])
print(output.reshape(-1,output.shape[-1]))
print(output.tolist())
out = []
out.append( torch.tensor([[1],[4],[7],[5]]))
out.append(torch.tensor([[1],[2],[3],[4]]))
print(out)
print(torch.cat(out,dim=1))