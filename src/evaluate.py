#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: evaluate.py
@date: 2025/10/27 16:57
@desc: 使用bleu评估，预测内容与test_target的关联程度
"""
import torch
from nltk.translate.bleu_score import corpus_bleu

from src import config
from src.dataset import get_dataloader
from src.model import TranslationSeq2Seq
from src.predict import predict_batch
from src.tokenizer import EnglishTokenizer, ChineseTokenizer


def evaluate(model, data_loader, en_tokenizer, device):
    model.eval()
    predictions = [] # 预测的
    references = [] # 生成的
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        # inputs.shape:[batch_size, seq_len]
        targets = targets.tolist() # 转换为python list
        # targets: [[],[],[seq_len]]
        batch_result = predict_batch(model, inputs,en_tokenizer=en_tokenizer)
        # [[gener_token],,] batch_size长度的有效token
        predictions.extend(batch_result) #预测
        references.extend( [[target[1:target.index(en_tokenizer.eos_token_index)]] for target in targets]) #为bleu提供参考
        # references = [
        #     [['I', 'love', 'you']],  # 第一句话的参考译文（可能有多个参考）
        #     [['This', 'is', 'a', 'book']],  # 第二句话的参考译文
        # ]
    print(references[0]) #list [[[7311, 5617, 6049, 27, 700, 1, 5864]],]
    print(predictions[0]) #list [[7311, 6049, 5454, 2918, 6375],]
    #id 转换为token
    pred_texts = [[en_tokenizer.index2word[index] for index in prediction] for prediction in predictions]
    refs_texts = []
    for reference in references: #[[7311, 5617, 6049, 27, 700, 1, 5864]]
        for target in reference: #[7311, 5617, 6049, 27, 700, 1, 5864]
            refs_text = [en_tokenizer.index2word[index] for index in target] #[hello,]
            refs_texts.append([refs_text])
    print(refs_texts[0])
    print(pred_texts[0])
    return corpus_bleu(refs_texts, pred_texts)

def run_evaluation():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 测试集
    test_dataloader = get_dataloader(train=False)
    #加载tokenizer
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    # 加载model
    model = TranslationSeq2Seq(zh_tokenizer.vocab_size, en_tokenizer.vocab_size,zh_tokenizer.pad_token_index,
                       en_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best_model.pt'))
    print('模型加载成功')


    #评估
    bleu = evaluate(model, test_dataloader, en_tokenizer, device)
    print(f'bleu: {bleu}')


if __name__ == '__main__':
    run_evaluation()