import json
import numpy as np
import argparse

from sentence_transformers import InputExample, SentenceTransformer, LoggingHandler
from sklearn.metrics.pairwise import cosine_similarity

def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model_path', type=str)
    parse.add_argument('--retrieval_extract_news', type=str)
    parse.add_argument('--ori_extract_news', type=str)
    parse.add_argument('--ori_train_mind_news', type=str)
    parse.add_argument('--ori_dev_mind_news', type=str)
    parse.add_argument('--train_mind_news', type=str)
    parse.add_argument('--dev_mind_news', type=str)
    parse.add_argument('--behavior_train_path', type=str)
    parse.add_argument('--behavior_test_path', type=str)
    parse.add_argument('--save_train_path', type=str)
    parse.add_argument('--save_test_path', type=str)
    args = parse.parse_args()
    return args

args = my_parse()

device = "cuda:0"
max_seq_length = 256
model_name = args.model_path
model = SentenceTransformer(model_name, device=device)
model.__setattr__("max_seq_length", max_seq_length)

file_path = args.retrieval_extract_news
wayback_extract_dataset_for_retrieval = json.load(open(file_path))
file_path = args.ori_extract_news
ori_wayback_extract_dataset = json.load(open(file_path))

mind_news_train_path = args.ori_train_mind_news
ori_mind_dataset = json.load(open(mind_news_train_path))
mind_news_dev_path = args.ori_dev_mind_news
ori_mind_dataset.extend(json.load(open(mind_news_dev_path)))

mind_re_news_train_path = args.train_mind_news
mind_dataset_for_retrieval = json.load(open(mind_re_news_train_path))
mind_re_news_dev_path = args.dev_mind_news
mind_dataset_for_retrieval.extend(json.load(open(mind_re_news_dev_path)))

ori_mind_wayback_dataset = ori_mind_dataset + ori_wayback_extract_dataset
mind_wayback_dataset = mind_dataset_for_retrieval + wayback_extract_dataset_for_retrieval

mind_emb = []
batch_size = 16
for i in range(0, len(mind_dataset_for_retrieval), batch_size):
    if i+batch_size > len(mind_dataset_for_retrieval):
        temp_data = mind_dataset_for_retrieval[i:len(mind_dataset_for_retrieval)]
    else:
        temp_data = mind_dataset_for_retrieval[i:i+batch_size]
    temp_mind_emb = model.encode(temp_data, batch_size=batch_size)
    mind_emb.extend(temp_mind_emb)

mind_emb = np.array(mind_emb)

extract_emb = []
batch_size = 16
for i in range(0, len(wayback_extract_dataset_for_retrieval), batch_size):
    if i+batch_size > len(wayback_extract_dataset_for_retrieval):
        temp_data = wayback_extract_dataset_for_retrieval[i:len(wayback_extract_dataset_for_retrieval)]
    else:
        temp_data = wayback_extract_dataset_for_retrieval[i:i+batch_size]
    temp_extract_emb = model.encode(temp_data, batch_size=batch_size)
    extract_emb.extend(temp_extract_emb)

extract_emb = np.array(extract_emb)

mind_wayback_emb = np.concatenate((mind_emb, extract_emb))

mind_wayback_cosine = cosine_similarity(mind_wayback_emb, mind_wayback_emb)

mind_wayback_id2index = {}
mind_wayback_index2id = {}
for index, temp in enumerate(ori_mind_wayback_dataset):
    mind_wayback_id2index[temp['id']] = index
    mind_wayback_index2id[str(index)] = temp['id']

behavior_path_list = [args.behavior_train_path, args.behavior_test_path]
save_path_list = [args.save_train_path, args.save_test_path]
for behavior_path, save_path in zip(behavior_path_list, save_path_list):
    behavior_dataset = json.load(open(
        behavior_path))
    behavior_with_event_news = []

    entity_name_2_emb = {}
    for imp in behavior_dataset:
        recommended_news = imp['clicked_news'][0]
        past_clicked_news = imp['past_clicked']

        event_news_index_list = np.where(0.8 < mind_wayback_cosine[mind_wayback_id2index[recommended_news]])[0]

        event_news_score_list = []
        for event_news_index in event_news_index_list:
            event_news_score_list.append(mind_wayback_cosine[mind_wayback_id2index[recommended_news]][event_news_index])
        event_news_index_sort = sorted(range(len(event_news_score_list)), key=lambda k: event_news_score_list[k],
                                         reverse=True)

        event_news_score_set = []
        event_news_index_set = set()
        for index in event_news_index_sort:
            if index in event_news_index_set:
                continue
            if event_news_index_list[index] == mind_wayback_id2index[recommended_news]:
                continue
            event_news_index_set.add(event_news_index_list[index])
            event_news_score_set.append(event_news_score_list[index])
        event_news_index_set = list(event_news_index_set)

        past_clicked_set = []
        for temp in past_clicked_news:
            past_clicked_set.append(mind_wayback_id2index[temp])

        past_clicked_content = []
        for news_id in imp['past_clicked']:
            past_clicked_content.append(ori_mind_wayback_dataset[mind_wayback_id2index[news_id]])

        clicked_news_content = []
        for news_id in imp['clicked_news']:
            clicked_news_content.append(ori_mind_wayback_dataset[mind_wayback_id2index[news_id]])

        temp_imp = {
            'id': imp['id'],
            'user_id': imp['user_id'],
            'datetime': imp['datetime'],
            'past_clicked': imp['past_clicked'],
            'past_clicked_content': past_clicked_content,
            'clicked_news': imp['clicked_news'],
            'clicked_news_content': clicked_news_content
        }
        if 'user_profile' in imp.keys():
            temp_imp['user_profile'] = imp['user_profile']

        event_news_id_dict = []
        event_news_id_set = set()
        event_news_content = []
        for i in range(len(event_news_index_set)):
            if mind_wayback_index2id[str(event_news_index_set[i])] in event_news_id_set:
                continue
            event_news_id_set.add(mind_wayback_index2id[str(event_news_index_set[i])])
            event_news_id_dict.append(mind_wayback_index2id[str(event_news_index_set[i])])
            event_news_content.append(
                ori_mind_wayback_dataset[event_news_index_set[i]]
            )

        temp_imp['event_news'] = event_news_id_dict
        temp_imp['event_news_content'] = event_news_content
        temp_imp.pop('non_clicked_news', None)

        if len(event_news_id_dict) < 2:
            continue

        behavior_with_event_news.append(temp_imp)

    json.dump(behavior_with_event_news, open(save_path, 'w'), indent=4)




