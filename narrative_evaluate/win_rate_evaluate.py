import sys
sys.path.append('/news_recommendation')
import numpy as np
import torch
import json
from config import PLM4NR_title_abstractConfig
from dataframe import read_behavior_df, read_news_df
from MINDDataset import MINDValDataset
from my_PLM4NR import my_PLM4NR
from NewsEncoder import NewsEncoder
from UserEncoder import UserEncoder
import csv
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, EvalPrediction
from transformers.modeling_outputs import ModelOutput
from utils.logger import logging
import argparse
import os
from utils.text import create_transform_fn_from_pretrained_tokenizer
from pathlib import Path

def evaluate(net: torch.nn.Module, eval_mind_dataset: MINDValDataset,
             device: torch.device) -> RecMetrics:
    net.eval()
    EVAL_BATCH_SIZE = 1

    eval_dataloader = DataLoader(eval_mind_dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=True, shuffle=False)
    fhit_score = 0
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            model_output: ModelOutput = net(**batch)

        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()[:2]

        if y_score[0] > y_score[1]:
            fhit_score += 1
    fhit_score = fhit_score / len(eval_mind_dataset)
    return fhit_score

def pred_to_tsv(args, test_file):
    save_dir = './data/temp_save/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    behavior_tsv_save_path = os.path.join(save_dir, 'behaviors.tsv')
    with open(behavior_tsv_save_path, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for idx, imp in enumerate(test_file):
            tsv_writer.writerow([imp['id'], imp['user_id'], imp['datetime'],
                                 '; '.join([i['category'] + ', ' + ', '.join(i['topic']) for i in imp['user_profile']]),
                                 ' '.join(imp['past_clicked']),
                                 'P' + str(idx) + '-1 C'+str(idx)+'-0'])

    news_tsv_save_path = os.path.join(save_dir, 'news.tsv')
    with open(news_tsv_save_path, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for idx, imp in enumerate(test_file):
            for past_clicked_idx, past_clicked in zip(imp['past_clicked'], imp['past_clicked_content']):
                tsv_writer.writerow([past_clicked_idx,
                                     past_clicked['category'],
                                     'news',
                                     ', '.join(past_clicked['topic']),
                                     past_clicked['title'],
                                     past_clicked['abstract'],
                                     'Title Entities', 'Abstract Entites'])
            tsv_writer.writerow(['C'+str(idx),
                                 imp['clicked_news_content'][0]['category'],
                                 'news',
                                 ', '.join(imp['clicked_news_content'][0]['topic']),
                                 imp['clicked_news_content'][0]['title'],
                                 imp['clicked_news_content'][0]['abstract'],
                                 'Title Entities', 'Abstract Entites'])
            tsv_writer.writerow(['P' + str(idx),
                                 imp['personalized_news']['category'],
                                 'news',
                                 imp['personalized_news']['topic'],
                                 imp['personalized_news']['title'],
                                 imp['personalized_news']['abstract'],
                                 'Title Entities', 'Abstract Entites'])
    return behavior_tsv_save_path, news_tsv_save_path

def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_file_path', type=str)
    parse.add_argument('--checkpoint_dir', type=str, default=None)
    args = parse.parse_args()
    return args


if __name__ == "__main__":
    config = PLM4NR_title_abstractConfig()
    args = my_parse()

    test_file = json.load(open(args.test_file_path))
    behavior_tsv_save_path, news_tsv_save_path = pred_to_tsv(args, test_file)

    hidden_size: int = AutoConfig.from_pretrained(config.pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(
        AutoTokenizer.from_pretrained(config.pretrained),
        train_mode = False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    news_encoder = NewsEncoder(config.pretrained)
    interest_encoder = NewsEncoder(config.pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    plm4nr_net = my_PLM4NR(news_encoder=news_encoder, interest_encoder=interest_encoder,
                           user_encoder=user_encoder,
                           hidden_size=hidden_size, loss_fn=loss_fn).to(
        device, dtype=torch.bfloat16
    )
    plm4nr_net.load_state_dict(torch.load(args.checkpoint_dir))

    test_news_df = read_news_df(news_tsv_save_path, clear_cache=True)
    test_behavior_df = read_behavior_df(behavior_tsv_save_path, clear_cache=True)
    test_dataset = MINDValDataset(test_behavior_df, test_news_df, transform_fn, config.history_size)

    hit_score = evaluate(plm4nr_net, test_dataset, device)

    logging.info({'Win Rate': hit_score})
