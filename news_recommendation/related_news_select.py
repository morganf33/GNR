import torch
import json
from config import PLM4NR_title_abstractConfig
from RecEvaluator import RecMetrics
from dataframe import read_behavior_df, read_news_df
from MINDDataset import MINDValDataset
from my_PLM4NR import my_PLM4NR
from NewsEncoder import NewsEncoder
from UserEncoder import UserEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from utils.logger import logging
from utils.text import create_transform_fn_from_pretrained_tokenizer
from typing import List, Dict
import argparse
import csv
from pathlib import Path

def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--checkpoint_file', type=str)
    parse.add_argument('--behavior_train_file', type=str)
    parse.add_argument('--train_save_path', type=str)
    parse.add_argument('--behavior_test_file', type=str)
    parse.add_argument('--test_save_path', type=str)
    args = parse.parse_args()
    return args

def pred_to_tsv(behavior_file):
    pred_behavior = json.load(open(behavior_file))
    past_idx_set = set()
    event_idx_set = set()
    news_tsv_save_path = './data/temp_save/news.tsv'
    Path('./data/temp_save').mkdir(parents=True, exist_ok=True)
    with open(news_tsv_save_path, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for idx, imp in enumerate(pred_behavior):
            for past_clicked_idx, past_clicked in zip(imp['past_clicked'], imp['past_clicked_content']):
                if past_clicked_idx not in past_idx_set:
                    past_idx_set.add(past_clicked_idx)
                    tsv_writer.writerow([past_clicked_idx,
                                         past_clicked['category'],
                                         'news',
                                         ', '.join(past_clicked['topic']),
                                         past_clicked['title'],
                                         past_clicked['abstract'],
                                         'Title Entities', 'Abstract Entites'])
            for event_news_idx, event_news in zip(imp['event_news'], imp['event_news_content']):
                if event_news_idx not in event_idx_set:
                    past_idx_set.add(event_news_idx)
                    tsv_writer.writerow([event_news_idx,
                                         'politics',
                                         'news',
                                         ', '.join(event_news['topic']),
                                         event_news['title'],
                                         event_news['abstract'],
                                         'Title Entities', 'Abstract Entites'])
                    
    behavior_tsv_save_path = './data/temp_save/behaviors.tsv'
    with open(behavior_tsv_save_path, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for idx, imp in enumerate(pred_behavior):
            tsv_writer.writerow([imp['id'], imp['user_id'], imp['datetime'],
                                 ', '.join(imp['user_profile']),
                                 '; '.join([i['category'] + ', ' + ', '.join(i['topic']) for i in imp['user_profile']]),
                                 ' '.join([i+'-0' for i in imp['event_news']])
                                 ])
    return news_tsv_save_path, behavior_tsv_save_path

def ranking(net: torch.nn.Module, eval_mind_dataset: MINDValDataset, device: torch.device, ori_test_dataset:List[Dict]):
    net.eval()
    EVAL_BATCH_SIZE = 1
    eval_dataloader = DataLoader(eval_mind_dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=True, shuffle=False)

    behaviors_with_related_news = []
    count = 0
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            model_output: ModelOutput = net(**batch)
        count += 1

        y_pred: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()

        related_mix_score_index_set = sorted(range(len(y_pred)),
                                             key=lambda k: y_pred[k], reverse=True)

        related_news_content_set = []
        related_news_idx_set = set()
        for index in related_mix_score_index_set[:-1]:
            if len(related_news_content_set) >= 3:
                break
            else:
                temp_dict = ori_test_dataset[count - 1]['event_news_content'][index]

                if temp_dict['id'] in related_news_idx_set or \
                        temp_dict['id'] in ori_test_dataset[count - 1]['past_clicked'] or \
                        temp_dict['id'] in ori_test_dataset[count - 1]['clicked_news']:
                    continue
                related_news_idx_set.add(temp_dict['id'])
                temp_dict['score'] = y_pred[index]
                temp_dict.pop('related_news_content', None)
                temp_dict.pop('related_link', None)
                related_news_content_set.append(temp_dict)

        ori_test_dataset[count - 1].pop('event_news', None)
        ori_test_dataset[count - 1].pop('event_news_content', None)
        ori_test_dataset[count - 1]['related_news'] = list(related_news_idx_set)
        ori_test_dataset[count - 1]['related_news_content'] = related_news_content_set
        behaviors_with_related_news.append(ori_test_dataset[count - 1])
    return behaviors_with_related_news


if __name__ == "__main__":
    config = PLM4NR_title_abstractConfig()
    args = my_parse()
    news_tsv_save_path, behavior_tsv_save_path = pred_to_tsv(args.behavior_train_file)
    hidden_size: int = AutoConfig.from_pretrained(config.pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(config.pretrained), train_mode=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Initialize Model")
    news_encoder = NewsEncoder(config.pretrained)
    topics_encoder = NewsEncoder(config.pretrained)
    interest_encoder = NewsEncoder(config.pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    plm4nr_net = my_PLM4NR(news_encoder=news_encoder, topics_encoder=topics_encoder,
                                interest_encoder=interest_encoder, user_encoder=user_encoder,
                                hidden_size=hidden_size, loss_fn=loss_fn).to(
        device, dtype=torch.bfloat16
    )
    plm4nr_net.load_state_dict(torch.load(args.checkpoint_file))

    test_news_df = read_news_df(news_tsv_save_path, clear_cache=True)
    test_behavior_df = read_behavior_df(behavior_tsv_save_path, clear_cache=True)
    test_dataset = MINDValDataset(test_behavior_df, test_news_df, transform_fn, config.history_size)

    ori_test_path = json.load(open(args.behavior_train_file))
    behaviors_with_related_news = ranking(plm4nr_net, test_dataset, device, ori_test_path)

    json.dump(behaviors_with_related_news, open(args.train_save_path, 'w'), indent=4)

    news_tsv_save_path, behavior_tsv_save_path = pred_to_tsv(args.behavior_test_file)
    hidden_size: int = AutoConfig.from_pretrained(config.pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(
        AutoTokenizer.from_pretrained(config.pretrained), train_mode=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_news_df = read_news_df(news_tsv_save_path, clear_cache=True)
    test_behavior_df = read_behavior_df(behavior_tsv_save_path, clear_cache=True)
    test_dataset = MINDValDataset(test_behavior_df, test_news_df, transform_fn, config.history_size)

    ori_test_path = json.load(open(args.behavior_test_file))
    behaviors_with_related_news = ranking(plm4nr_net, test_dataset, device, ori_test_path)

    json.dump(behaviors_with_related_news, open(args.test_save_path, 'w'), indent=4)