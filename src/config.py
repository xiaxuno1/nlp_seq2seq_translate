#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: config.py
@date: 2025/10/23 15:09
@desc:基本的配置路径，超参数
"""
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent # 根目录
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32  #64过大会退出
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 20

