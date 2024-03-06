import json
import os.path

import pandas as pd
import random
import argparse
from pathlib import Path


def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--ori_dataset', type=str)
    parse.add_argument('--training_dataset', type=str)
    parse.add_argument('--testing_dataset', type=str)
    parse.add_argument('--ori_extract_news', type=str)
    parse.add_argument('--extract_news', type=str)
    parse.add_argument('--ori_mind_news', type=list)
    parse.add_argument('--mind_news', type=list)
    args = parse.parse_args()
    return args

args = my_parse()
dataset = json.load(open(args.ori_dataset))
id_2_news = {}

for news in dataset:
    id_2_news[news['id']] = news
    for r_news in news['related_news_content']:
        id_2_news[r_news['id']] = r_news

sent_0 = []
sent_1 = []
sent_2 = []
max_len = 256

for data in dataset:
    if 'related_news_content' not in data.keys():
        continue
    temp_0 = data['title'] + '; ' + data['category'] + '; ' + ', '.join(data['topic']) + '; ' + data['abstract']

    ref_idx_set = set()
    for ref in data['related_news_content']:
        ref_idx_set.add(ref['id'])

    for ref in data['related_news_content']:
        ref_news = id_2_news[ref['id']]
        temp_1 = ref_news['title'] + '; ' + ref_news['category'] + '; ' + ', '.join(ref_news['topic']) + '; ' + ref_news['abstract']

        for _ in range(1):
            randi = random.randint(0, len(id_2_news)-1)
            rand_id = list(id_2_news.keys())[randi]
            if rand_id in ref_idx_set or rand_id==data['id']:
                continue
            temp_2 = id_2_news[rand_id]['title'] + '; ' + id_2_news[rand_id]['category'] + '; ' + ', '.join(id_2_news[rand_id]['topic']) + '; ' + id_2_news[rand_id]['abstract']
            sent_2.append(temp_2)

        for _ in range(1):
            sent_0.append(temp_0)
            sent_1.append(temp_1)

eval_idx = []
while(len(eval_idx)<100):
    eval_idx.append(random.randint(1, len(sent_0)))
    eval_idx = list(set(eval_idx))

eval_sent_0 = []
eval_sent_1 = []
eval_score = []
train_sent_0 = []
train_sent_1 = []
train_sent_2 = []
train_score = []
for idx in range(len(sent_0)):
    if idx in eval_idx:
        eval_sent_0.append(sent_0[idx])
        eval_sent_1.append(sent_1[idx])
        eval_score.append(5)
        for j in range(3):
            eval_sent_0.append(sent_0[idx])
            eval_sent_1.append(sent_1[random.randint(0, len(sent_1)-1)])
            eval_score.append(0)
    else:
        train_sent_0.append(sent_0[idx])
        train_sent_1.append(sent_1[idx])
        train_score.append(5)
        for j in range(3):
            train_sent_0.append(sent_0[idx])
            train_sent_1.append(sent_1[random.randint(0, len(sent_0) - 1)])
            train_score.append(0)

dataframe = pd.DataFrame({'sent0': train_sent_0, 'sent1': train_sent_1, 'score': train_score})
dataframe.to_csv(args.training_dataset, index=False, sep=',', header=None)

dataframe = pd.DataFrame({'sent0': eval_sent_0, 'sent1': eval_sent_1, 'score': eval_score})
dataframe.to_csv(args.testing_dataset, index=False, sep=',', header=None)


news_with_topic = json.load(open(args.ori_mind_news[0]))
news_4_sbert = []
for data in news_with_topic:
    temp = data['title'] + '; ' + data['category'] + '; ' + ', '.join(data['topic']) + '; ' + data['abstract']
    news_4_sbert.append(temp)
json.dump(news_4_sbert, open(args.mind_news[0], 'w'), indent=4)

news_with_topic = json.load(open(args.ori_mind_news[1]))
news_4_sbert = []
for data in news_with_topic:
    temp = data['title'] + '; ' + data['category'] + '; ' + ', '.join(data['topic']) + '; ' + data['abstract']
    news_4_sbert.append(temp)
json.dump(news_4_sbert, open(args.mind_news[1], 'w'), indent=4)

news_with_topic = json.load(open(args.ori_extract_news))
news_4_sbert = []
for data in news_with_topic:
    temp = data['title'] + '; ' + data['category'] + '; ' + ', '.join(data['topic']) + '; ' + data['abstract']
    news_4_sbert.append(temp)
json.dump(news_4_sbert, open(args.extract_news, 'w'), indent=4)




