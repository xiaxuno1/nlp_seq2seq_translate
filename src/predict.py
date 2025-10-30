#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: predict.py
@date: 2025/10/27 19:00
@desc:预测推理
"""
import torch

from src import config
from src.model import TranslationSeq2Seq
from src.tokenizer import ChineseTokenizer, EnglishTokenizer


def predict_batch(model, inputs, en_tokenizer):
    model.eval()
    with torch.no_grad():
        # inputs:[batch_size,seq_len]
        context_vector = model.encoder(inputs) # 编码器，得到上下文向量，用作decoder的input
        # context_vector:[batch_size,hidden_size]

        #解码
        batch_size = inputs.size(0)
        device = inputs.device

        #隐藏状态
        decoder_hidden = context_vector.unsqueeze(0) #因为作为输入少了一个维度，因此在第0轴增加一个维度
        # [1,batch_size,hidden_size]
        decoder_input = torch.full([batch_size, 1], en_tokenizer.sos_token_index,device=device)
        # [[eos_token_index],[eos],,,,] 创建一个size大小的tensor,以fill_value填充，dtype为fill_value类型

        #预测结果缓存
        generated = []
        # [batch_sizeM，MAX_SEQ_LENGTH]

        #记录每个样本是否已经生成结束符
        is_finished = torch.full([batch_size], False, device=device)
        # tensor:[batch_size] ,bool类型

        #自回归生成，每个输出预测作为，下个token的输入
        for i in range(config.MAX_SEQ_LENGTH): #设置最大生成长度
            #解码
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            # [B,1,V]       [1,B,H]

            #保存预测结果
            next_token_indexes = torch.argmax(decoder_output, dim=-1) #降维：[B,1] 得到batch_size的一个预测结果使用（max）
            generated.append(next_token_indexes)
            #更新输入，输出作为输入
            decoder_input = next_token_indexes
            #判段是否应该结束 next_token_indexes.squeeze(1)把[B,1]变为[B]
            is_finished |= (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
            if is_finished.all(): #所有样本都生成完毕[True,True,,,,]
                break
        #处理预测结果
        #整理预测结果形状generated：[batch_size，MAX_SEQ_LENGTH]
        generated_tensor = torch.cat(generated, dim=1) #按照1轴降维[[MAX_SEQ_LENGTH]]
        generated_list = generated_tensor.tolist()
        #去掉eos之后的token id
        for index, sentence in enumerate(generated_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]
        return generated_list

def predict(text, model, zh_tokenizer, en_tokenizer, device):
    indexes = zh_tokenizer.encode(text)
    input_tensor = torch.tensor([indexes], device=device,dtype=torch.long)

    #预测
    batch_result = predict_batch(model, input_tensor, en_tokenizer)
    return en_tokenizer.decode(batch_result[0])

def run_predict():
    # 准备资源
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2.分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    print("分词器加载成功")

    # 3. 模型
    model = TranslationSeq2Seq(zh_tokenizer.vocab_size, en_tokenizer.vocab_size, zh_tokenizer.pad_token_index,
                             en_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best_model.pt'))
    print("模型加载成功")

    print("欢迎使用中英翻译模型(输入q或者quit退出)")

    while True:
        user_input = input("中文：")
        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break
        if user_input.strip() == '':
            print("请输入内容")
            continue

        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print("英文：", result)


if __name__ == '__main__':
    run_predict()