import json
import csv
import argparse

def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--input_train_dev_news', type=str)
    parse.add_argument('--input_train_behaviors', type=str)
    parse.add_argument('--input_dev_behaviors', type=str)
    parse.add_argument('--input_test_news', type=str)
    parse.add_argument('--input_test_behaviors', type=str)
    parse.add_argument('--output_train_dev_news', type=str)
    parse.add_argument('--output_train_behaviors', type=str)
    parse.add_argument('--output_dev_behaviors', type=str)
    parse.add_argument('--output_test_news', type=str)
    parse.add_argument('--output_test_behaviors', type=str)
    args = parse.parse_args()
    return args

args = my_parse()
train_dev_news_dataset = json.load(open(args.input_train_dev_news))
train_behavior_dataset = json.load(open(args.input_train_behaviors))
dev_behavior_dataset = json.load(open(args.input_dev_behaviors))

test_news_dataset = json.load(open(args.input_test_news))
test_behavior_dataset = json.load(open(args.input_test_behaviors))

with open(args.output_train_dev_news, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for idx, news in enumerate(train_dev_news_dataset):
        tsv_writer.writerow([news['id'],
                             news['category'],
                             'news',
                             # ', '.join(news['topic']),
                             news['title'],
                             news['abstract'],
                             'Title Entities', 'Abstract Entites'])


with open(args.output_test_news, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for idx, news in enumerate(test_news_dataset):
        tsv_writer.writerow([news['id'],
                             news['category'],
                             'news',
                             # ', '.join(news['topic']),
                             news['title'],
                             news['abstract'],
                             'Title Entities', 'Abstract Entites'])

with open(args.output_train_behaviors, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for idx, imp in enumerate(train_behavior_dataset):
        non_clicked = []
        for news in imp['non_clicked_news']:
            non_clicked.append(news)
        past_clicked = []
        for news in imp['past_clicked']:
            past_clicked.append(news)
        clicked_string = '-1 '.join(imp['clicked_news']) + '-1 ' + '-0 '.join(non_clicked) + '-0'
        tsv_writer.writerow([imp['id'], imp['user_id'],
                             imp['datetime'],
                             ', '.join(imp['user_profile']),
                             ' '.join(past_clicked),
                             clicked_string])

with open(args.output_dev_behaviors, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for idx, imp in enumerate(dev_behavior_dataset):
        non_clicked = []
        for news in imp['non_clicked_news']:
            non_clicked.append(news)
        past_clicked = []
        for news in imp['past_clicked']:
            if news in past_clicked:
                continue
            past_clicked.append(news)

        clicked_string = '-1 '.join(imp['clicked_news']) + '-1 ' + '-0 '.join(non_clicked) + '-0'
        tsv_writer.writerow([imp['id'], imp['user_id'],
                             imp['datetime'],
                             ', '.join(imp['user_profile']),
                             ' '.join(past_clicked),
                             clicked_string])

with open(args.output_test_behaviors, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for idx, imp in enumerate(test_behavior_dataset):
        non_clicked = []
        for news in imp['non_clicked_news']:
            non_clicked.append(news)
        past_clicked = []
        for news in imp['past_clicked']:
            if news in past_clicked:
                continue
            past_clicked.append(news)

        clicked_string = '-1 '.join(imp['clicked_news']) + '-1 ' + '-0 '.join(non_clicked) + '-0'
        tsv_writer.writerow([imp['id'], imp['user_id'],
                             imp['datetime'],
                             ', '.join(imp['user_profile']),
                             ' '.join(past_clicked),
                             clicked_string])

