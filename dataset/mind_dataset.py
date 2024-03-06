import json
import csv
from typing import List
from dataclasses import dataclass
import tqdm as tq
from pathlib import Path


file_path = '/Users/jiabaofang/PycharmProjects/pythonProject1/mind/MINDlarge_train/news.tsv'
csv_reader = csv.reader(open(file_path), delimiter='\t')

id_2_news = {}
ori_dataset = []
news_idx = json.load(open('./recsys/mind_train/news_idx.json'))
for idx, row in enumerate(csv_reader):
    if row[0] in news_idx:
        temp = {'id': row[0], 'category': 'politics', 'title': row[3], 'abstract': row[4]}
        ori_dataset.append(temp)
        id_2_news[row[0]] = temp

save_file = './recsys/mind_train/news.json'
json.dump(ori_dataset, open(save_file, 'w'), indent=4)

save_file = './generator/mind_train/news.json'
json.dump(ori_dataset, open(save_file, 'w'), indent=4)


file_path = '/Users/jiabaofang/PycharmProjects/pythonProject1/mind/MINDlarge_dev/news.tsv'
csv_reader = csv.reader(open(file_path), delimiter='\t')

id_2_news = {}
ori_dataset = []
news_idx = json.load(open('./recsys/mind_test/news_idx.json'))
for idx, row in enumerate(csv_reader):
    if row[0] in news_idx:
        temp = {'id': row[0], 'category': 'politics', 'title': row[3], 'abstract': row[4]}
        ori_dataset.append(temp)
        id_2_news[row[0]] = temp

save_file = './recsys/mind_test/news.json'
json.dump(ori_dataset, open(save_file, 'w'), indent=4)

save_file = './generator/mind_test/news.json'
json.dump(ori_dataset, open(save_file, 'w'), indent=4)

tsv_file = open("/Users/jiabaofang/PycharmProjects/pythonProject1/mind/MINDlarge_dev/behaviors.tsv", "r")
read_tsv = csv.reader(tsv_file, delimiter="\t")

@dataclass
class Impression:
    id: str
    user_id: str
    datetime: str
    past_clicked: List
    clicked_news: List
    non_clicked_news: List


impressions = []
imp_idx = json.load(open('./recsys/mind_test/behaviors_idx.json'))
for row in tq.tqdm(read_tsv):
    if str(row[0]) not in imp_idx:
        continue
    imp = Impression(id=row[0], user_id=row[1], datetime=row[2], past_clicked=row[3].split(),
                     clicked_news=[i.split("-")[0] for i in row[-1].split() if i.endswith("-1")],
                     non_clicked_news=[i.split("-")[0] for i in row[-1].split() if i.endswith("-0")])

    clicked_news_select = set()
    for clicked_news in imp.clicked_news:
        if clicked_news in news_idx:
            clicked_news_select.add(clicked_news)

    past_clicked_select = set()
    non_clicked_select = set()

    for past in imp.past_clicked:
        if past in news_idx:
            past_clicked_select.add(past)
    for non_clicked in imp.non_clicked_news:
        if non_clicked in news_idx:
            non_clicked_select.add(non_clicked)

    impressions.append({'id': imp.id, 'user_id': imp.user_id, 'datetime': imp.datetime,
                        'past_clicked': list(past_clicked_select), 'clicked_news': list(clicked_news_select),
                        'non_clicked_news': list(non_clicked_select)})

save_file = './recsys/mind_test/behaviors.json'
json.dump(impressions, open(save_file, 'w'), indent=4)

tsv_file = open("/Users/jiabaofang/PycharmProjects/pythonProject1/mind/MINDlarge_dev/behaviors.tsv", "r")
read_tsv = csv.reader(tsv_file, delimiter="\t")
impressions = []
imp_idx = json.load(open('./generator/mind_test/behaviors_idx.json'))
for row in tq.tqdm(read_tsv):
    if str(row[0]) not in imp_idx:
        continue
    imp = Impression(id=row[0], user_id=row[1], datetime=row[2], past_clicked=row[3].split(),
                     clicked_news=[i.split("-")[0] for i in row[-1].split() if i.endswith("-1")],
                     non_clicked_news=[i.split("-")[0] for i in row[-1].split() if i.endswith("-0")])

    clicked_news_select = set()
    for clicked_news in imp.clicked_news:
        if clicked_news in news_idx:
            clicked_news_select.add(clicked_news)

    past_clicked_select = set()

    for past in imp.past_clicked:
        if past in news_idx:
            past_clicked_select.add(past)

    impressions.append({'id': imp.id, 'user_id': imp.user_id, 'datetime': imp.datetime,
                        'past_clicked': list(past_clicked_select), 'clicked_news': list(clicked_news_select)})

save_file = './generator/mind_test/behaviors.json'
json.dump(impressions, open(save_file, 'w'), indent=4)


tsv_file = open("/Users/jiabaofang/PycharmProjects/pythonProject1/mind/MINDlarge_train/behaviors.tsv", "r")
read_tsv = csv.reader(tsv_file, delimiter="\t")
news_idx = json.load(open('./recsys/mind_train/news_idx.json'))

train_impressions = []
dev_impressions = []
train_imp_idx = json.load(open('./recsys/mind_train/behaviors_train_idx.json'))
dev_imp_idx = json.load(open('./recsys/mind_train/behaviors_val_idx.json'))
for row in tq.tqdm(read_tsv):
    if str(row[0]) not in train_imp_idx and str(row[0]) not in dev_imp_idx:
        continue
    imp = Impression(id=row[0], user_id=row[1], datetime=row[2], past_clicked=row[3].split(),
                     clicked_news=[i.split("-")[0] for i in row[-1].split() if i.endswith("-1")],
                     non_clicked_news=[i.split("-")[0] for i in row[-1].split() if i.endswith("-0")])

    clicked_news_select = set()
    for clicked_news in imp.clicked_news:
        if clicked_news in news_idx:
            clicked_news_select.add(clicked_news)

    past_clicked_select = set()
    non_clicked_select = set()

    for past in imp.past_clicked:
        if past in news_idx:
            past_clicked_select.add(past)
    for non_clicked in imp.non_clicked_news:
        if non_clicked in news_idx:
            non_clicked_select.add(non_clicked)

    if imp.id in dev_imp_idx:
        dev_impressions.append({'id': imp.id, 'user_id': imp.user_id, 'datetime': imp.datetime,
                            'past_clicked': list(past_clicked_select), 'clicked_news': list(clicked_news_select),
                            'non_clicked_news': list(non_clicked_select)})
    elif imp.id in train_imp_idx:
        train_impressions.append({'id': imp.id, 'user_id': imp.user_id, 'datetime': imp.datetime,
                                'past_clicked': list(past_clicked_select), 'clicked_news': list(clicked_news_select),
                                'non_clicked_news': list(non_clicked_select)})

save_file = './recsys/mind_train/behaviors_train.json'
json.dump(train_impressions, open(save_file, 'w'), indent=4)

save_file = './recsys/mind_train/behaviors_val.json'
json.dump(dev_impressions, open(save_file, 'w'), indent=4)

tsv_file = open("/Users/jiabaofang/PycharmProjects/pythonProject1/mind/MINDlarge_train/behaviors.tsv", "r")
read_tsv = csv.reader(tsv_file, delimiter="\t")
news_idx = json.load(open('./recsys/mind_train/news_idx.json'))

impressions = []
imp_idx = json.load(open('./generator/mind_train/behaviors_idx.json'))
for row in tq.tqdm(read_tsv):
    if str(row[0]) not in imp_idx:
        continue
    imp = Impression(id=row[0], user_id=row[1], datetime=row[2], past_clicked=row[3].split(),
                     clicked_news=[i.split("-")[0] for i in row[-1].split() if i.endswith("-1")],
                     non_clicked_news=[i.split("-")[0] for i in row[-1].split() if i.endswith("-0")])

    clicked_news_select = set()
    for clicked_news in imp.clicked_news:
        if clicked_news in news_idx:
            clicked_news_select.add(clicked_news)

    past_clicked_select = set()

    for past in imp.past_clicked:
        if past in news_idx:
            past_clicked_select.add(past)

    impressions.append({'id': imp.id, 'user_id': imp.user_id, 'datetime': imp.datetime,
                        'past_clicked': list(past_clicked_select), 'clicked_news': list(clicked_news_select)})

save_file = './generator/mind_train/behaviors.json'
json.dump(impressions, open(save_file, 'w'), indent=4)
