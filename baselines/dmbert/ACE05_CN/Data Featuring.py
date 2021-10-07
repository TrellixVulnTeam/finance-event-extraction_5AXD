# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 10:44:57 2021

@author: David Xu
"""
import random
from transformers import BertTokenizer

# 指定繁简中文 BERT-BASE预训练模型
PRETRAINED_MODEL_NAME = "bert-base-chinese"
# 获取预测模型所使用的tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
sets = ["train", "test", "dev"]
for s in sets:
    with open("C:/Users/David Xu/PycharmProjects/Doc2EDAG/Data/{}.json".format(s), "r", encoding=("utf-8")) as f:
        lines = f.readlines()

    string = "".join(lines)
    string = string.replace("null", "None")
    # 解析string
    docs = (eval(string))

    dataset = []
    for i in range(len(docs)):
        doc = docs[i][1]
        data = {}
        entites = "recguid_eventname_eventdict_list"
        idx = random.randint(0, len(doc[entites])-1)

        # random to select a event
        data["event_type"] = doc[entites][idx][1]

        # select a trigger according to TradedShared, EquityHolder, or Others
        if "TradedShares" in doc[entites][idx][2]:
            token = doc[entites][idx][2]["TradedShares"]
        elif 'EquityHolder' in doc[entites][idx][2]:
            token = doc[entites][idx][2]["EquityHolder"]
        else:
            keys = list(doc[entites][idx][2].keys())
            temp_keys = []
            for key in keys:
                if doc[entites][idx][2][key]:
                    temp_keys.append(key)
            temp = random.randint(0, len(temp_keys)-1)
            token = doc[entites][idx][2][temp_keys[temp]]

        data["trigger_tokens"] = tokenizer.tokenize(token)
        token_range = doc["ann_mspan2dranges"][token][0]
        data["trigger_start"] = token_range[1]
        data["trigger_end"] = token_range[2]
        data["tokens"] = tokenizer.tokenize(doc["sentences"][token_range[0]])

        dataset.append(data)

    # output
    f = open("{}_CNFin.json".format(s), "w", encoding='utf-8')
    f.write(str(dataset))
    f.close()
