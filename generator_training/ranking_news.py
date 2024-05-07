import sys
sys.path.append('/news_recommendation')
import torch
import json
import argparse
from config import PLM4NR_title_abstractConfig
from dataframe import read_behavior_df, read_news_df
from MINDDataset import MINDValDataset
from my_PLM4NR import my_PLM4NR
from NewsEncoder import NewsEncoder
from UserEncoder import UserEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, EvalPrediction
from transformers.modeling_outputs import ModelOutput
from utils.logger import logging
from utils.text import create_transform_fn_from_pretrained_tokenizer
from typing import List, Dict
import re
import csv
from pathlib import Path


def extract_info(s):
    pattern_list = [r'"title":\s*"([^"]+)"', r'"title":\s*([^"]+)', r'"title" :\s*"([^"]+)"']
    for pattern in pattern_list:
        match = re.search(pattern, s)
        if match is not None:
            break
    try:
        title = match.group(1)
    except BaseException:
        return None

    pattern_list = [r'"category":\s*"([^"]+)"', r'"category":\s*([^"]+)', r'"category" :\s*"([^"]+)"']
    for pattern in pattern_list:
        match = re.search(pattern, s)
        if match is not None:
            break
    try:
        category = match.group(1)
    except BaseException:
        return None

    pattern_list = [r'"topic":\s*"([^"]+)"', r'"topics":\s*"([^"]+)"', r'"topic" :\s*"([^"]+)"',
                    r'"topics" :\s*"([^"]+)"', r'" topics":\s*"([^"]+)"']
    for pattern in pattern_list:
        match = re.search(pattern, s)
        if match is not None:
            break
    if match is None:
        return None
    topic = match.group(1)

    pattern_list = [r'"abstract":\s*"([^"]+)"', r'" abstract":\s*"([^"]+)"', r'"abstract":\s*"([^"]+)',
                    r'"abstract" :\s*"([^"]+)"']
    for pattern in pattern_list:
        match = re.search(pattern, s)
        if match is not None:
            break
    if match is None:
        return None
    try:
        abstract = match.group(1)
    except BaseException:
        return None

    return {'title': title, 'category': category, 'topic': topic, 'abstract': abstract}

def process_index(news_index, imp):
    if news_index == 0:
        return imp['prediction']
    elif news_index == 1:
        return imp['clicked_news_content'][0]
    elif news_index == 2:
        return imp['personalized_news']
    elif news_index > 2:
        return None
    else:
        raise AttributeError

def ranking(net: torch.nn.Module, eval_mind_dataset: MINDValDataset, device: torch.device):
    net.eval()
    EVAL_BATCH_SIZE = 1

    eval_dataloader = DataLoader(eval_mind_dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=True, shuffle=False)
    output_index = []
    output_index_2_score = []
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            model_output: ModelOutput = net(**batch)
        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()

        sorted_y_score = sorted(range(len(y_score)), key=lambda k: y_score[k], reverse=True)
        output_index.append(sorted_y_score)
        sorted_y_score_dict = {}
        for index in sorted_y_score:
            sorted_y_score_dict[index] = y_score[index]
        output_index_2_score.append(sorted_y_score_dict)

    return output_index, output_index_2_score

def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--ori_test_path', type=str)
    parse.add_argument('--prediction_path', type=str)
    parse.add_argument('--save_path', type=str)
    parse.add_argument('--checkpoint_file', type=str)
    args = parse.parse_args()
    return args


def pred_process(args):
    ori_test = json.load(open(args.ori_test_path))
    post_test = []
    line_idx = 0
    with open(args.prediction_path, 'r', encoding="utf-8") as f:
        for line in f:
            temp = json.loads(line)['predict']
            ori_test[line_idx]['prediction'] = extract_info(temp)
            post_test.append(ori_test[line_idx])
            line_idx += 1
    return post_test

def pred_to_tsv(pred_behavior):
    news_tsv_save_path = './data/temp_save/news.tsv'
    Path('./data/temp_save').mkdir(parents=True, exist_ok=True)
    with open(news_tsv_save_path, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for idx, imp in enumerate(pred_behavior):
            for past_clicked_idx, past_clicked in zip(imp['past_clicked'], imp['past_clicked_content']):
                tsv_writer.writerow([past_clicked_idx,
                                     past_clicked['category'],
                                     'news',
                                     ', '.join(past_clicked['topic']),
                                     past_clicked['title'],
                                     past_clicked['abstract'],
                                     'Title Entities', 'Abstract Entites'])
            tsv_writer.writerow(['G' + str(idx),
                                 imp['clicked_news_content'][0]['category'],
                                 'news',
                                 ', '.join(imp['clicked_news_content'][0]['topic']),
                                 imp['clicked_news_content'][0]['title'],
                                 imp['clicked_news_content'][0]['abstract'],
                                 'Title Entities', 'Abstract Entites'])

            tsv_writer.writerow(['P' + str(idx),
                                 imp['prediction']['category'],
                                 'news',
                                 imp['prediction']['topic'],
                                 imp['prediction']['title'],
                                 imp['prediction']['abstract'],
                                 'Title Entities', 'Abstract Entites'])

            tsv_writer.writerow(['C' + str(idx),
                                 imp['prediction']['category'],
                                 'news',
                                 imp['personalized_news']['topic'],
                                 imp['personalized_news']['title'],
                                 imp['personalized_news']['abstract'],
                                 'Title Entities', 'Abstract Entites'])
    behavior_tsv_save_path = './data/temp_save/behaviors.tsv'
    with open(behavior_tsv_save_path, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for idx, imp in enumerate(pred_behavior):
            click_behavior = 'P' + str(idx) + '-1 G' + str(idx) + '-0 C' + str(idx) + '-0'

            tsv_writer.writerow([imp['id'], imp['user_id'], imp['datetime'],
                                 '; '.join([i['category'] + ', ' + ', '.join(i['topic']) for i in imp['user_profile']]),
                                 ' '.join(imp['past_clicked']),
                                 click_behavior])
    return news_tsv_save_path, behavior_tsv_save_path


if __name__ == "__main__":
    args = my_parse()

    config = PLM4NR_title_abstractConfig()
    post_test = pred_process(args)
    news_tsv_save_path, behavior_tsv_save_path = pred_to_tsv(post_test)

    hidden_size: int = AutoConfig.from_pretrained(config.pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(config.pretrained), train_mode=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    news_encoder = NewsEncoder(config.pretrained)
    interest_encoder = NewsEncoder(config.pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    plm4nr_net = my_PLM4NR(news_encoder=news_encoder, interest_encoder=interest_encoder, user_encoder=user_encoder,
                           hidden_size=hidden_size, loss_fn=loss_fn).to(
        device, dtype=torch.bfloat16
    )
    plm4nr_net.load_state_dict(torch.load(args.checkpoint_file))

    test_news_df = read_news_df(news_tsv_save_path, clear_cache=True)
    test_behavior_df = read_behavior_df(behavior_tsv_save_path, clear_cache=True)
    test_dataset = MINDValDataset(test_behavior_df, test_news_df, transform_fn, config.history_size)

    ranking_index_set, ranking_index_2_score_set = ranking(plm4nr_net, test_dataset, device)
    ori_test = json.load(open(args.ori_test_path))

    all_outputs = []

    instruction = """You are a personalized text generator. First, I will provide you with a news list that includes both the [main news] and [topic-related news]. Second, I will provide you with user interests, including the [categories] and [topics] of news that the user is interested in.
    Based on the input news list and user interests, you are required to generate a {personalized news summary} centered around the [main news].
    """
    for ranking_index, ranking_index_2_score, imp in zip(ranking_index_set, ranking_index_2_score_set, ori_test):
        temp_input = 'News List:\n[\n'
        temp_input = temp_input + '{"ID": "Main News", "title": "' + imp['clicked_news_content'][0]['title'] + \
               '", "category": "' + imp['clicked_news_content'][0]['category'] + \
               '", "topics": "' + ', '.join(imp['clicked_news_content'][0]['topic']) + \
               '", "abstract": "' + imp['clicked_news_content'][0]['abstract'] + '\n'

        for n_idx, related_news in enumerate(imp['related_news_content']):
            temp_input = temp_input + '{"ID": "Topic-related News ' + str(n_idx + 1) + \
                   '", "title": "' + related_news['title'] + \
                   '", "category": "' + related_news['category'] + \
                   '", "topics": "' + ', '.join(related_news['topic']) + \
                   '", "abstract": "' + related_news['abstract'] + '"},\n'

        temp_input = temp_input[:-2] + '\n'
        temp_input = temp_input + ']\n'

        temp_input = temp_input + 'User Interest:\n'
        for up in imp['user_profile']:
            temp_input = temp_input + 'This user is interested in news about [' + up['category'] + '], especially [' + ', '.join(
                up['topic']) + '].\n'

        responses = []
        scores = []
        for index in ranking_index:
            temp_news = process_index(index, imp)
            if temp_news is None:
                continue
            if isinstance(temp_news['topic'], List):
                temp_news_content = '{"title": "' + temp_news['title'] + \
                                   '", "category": "' + temp_news['category'] + \
                                   '", "topics": "' + ', '.join(temp_news['topic']) + \
                                   '", "abstract": "' + temp_news['abstract'] + '}\n'
            else:
                temp_news_content = '{"title": "' + temp_news['title'] + \
                                   '", "category": "' + temp_news['category'] + \
                                   '", "topics": "' + temp_news['topic'] + \
                                   '", "abstract": "' + temp_news['abstract'] + '}\n'

            responses.append(temp_news_content)
            scores.append(ranking_index_2_score[index])

        all_outputs.append({"query": instruction + temp_input, "responses": responses, "scores": scores})
    with open(args.save_path, 'w') as f:
        for item in all_outputs:
            f.write(json.dumps(item) + '\n')
