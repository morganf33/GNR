import json
import argparse
import os

def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--generate_file', type=str, default=None)
    parse.add_argument('--raw_dir', types=str, default=None)
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    args = my_parse()
    raw_dataset = json.load(open(args.raw_file))
    generate_dataset = json.load(open(args.generate_file))
    id_2_news = {}
    for news in generate_dataset:
        id_2_news[news['id']] = news

    news_dataset_path = os.path.join(args.raw_dir, 'generator/extract_news_with_related_news.json')
    save_path = os.path.join(args.raw_dir, 'generator/extract_news_with_related_news_ml.json')
    news_dataset = json.load(open(news_dataset_path))
    new_news_dataset = []
    for news in news_dataset:
        temp_news = id_2_news[news['id']]
        related_news_content = []
        if 'related_news_content' in list(news.keys()):
            for r_news in news['related_news_content']:
                related_news_content.append(id_2_news[r_news['id']])
        temp_news['related_news_content'] = related_news_content
        new_news_dataset.append(temp_news)
    json.dump(new_news_dataset, open(save_path, 'w'), indent=4)

    imp_dataset_path = os.path.join(args.raw_dir, 'recsys/mind_train/behaviors_train.json')
    save_path = os.path.join(args.raw_dir, 'recsys/mind_train/behaviors_train_nl.json')
    imp_dataset = json.load(open(imp_dataset_path))
    new_imp_dataset = []
    for imp in imp_dataset:
        past_clicked_content = []
        for news_idx in imp['past_clicked']:
            past_clicked_content.append(id_2_news[news_idx])
        imp['past_clicked_content'] = past_clicked_content
        clicked_news_content = []
        for news_idx in imp['clicked_news']:
            clicked_news_content.append(id_2_news[news_idx])
        imp['clicked_news_content'] = clicked_news_content
        new_imp_dataset.append(imp)
    json.dump(new_imp_dataset, open(save_path, 'w'), indent=4)

    imp_dataset_path = os.path.join(args.raw_dir, 'recsys/mind_train/behaviors_val.json')
    save_path = os.path.join(args.raw_dir, 'recsys/mind_train/behaviors_val_nl.json')
    imp_dataset = json.load(open(imp_dataset_path))
    new_imp_dataset = []
    for imp in imp_dataset:
        past_clicked_content = []
        for news_idx in imp['past_clicked']:
            past_clicked_content.append(id_2_news[news_idx])
        imp['past_clicked_content'] = past_clicked_content
        clicked_news_content = []
        for news_idx in imp['clicked_news']:
            clicked_news_content.append(id_2_news[news_idx])
        imp['clicked_news_content'] = clicked_news_content
        new_imp_dataset.append(imp)
    json.dump(new_imp_dataset, open(save_path, 'w'), indent=4)

    imp_dataset_path = os.path.join(args.raw_dir, 'generator/mind_train/behaviors.json')
    save_path = os.path.join(args.raw_dir, 'generator/mind_train/behaviors_nl.json')
    imp_dataset = json.load(open(imp_dataset_path))
    new_imp_dataset = []
    for imp in imp_dataset:
        past_clicked_content = []
        for news_idx in imp['past_clicked']:
            past_clicked_content.append(id_2_news[news_idx])
        imp['past_clicked_content'] = past_clicked_content
        clicked_news_content = []
        for news_idx in imp['clicked_news']:
            clicked_news_content.append(id_2_news[news_idx])
        imp['clicked_news_content'] = clicked_news_content
        new_imp_dataset.append(imp)
    json.dump(new_imp_dataset, open(save_path, 'w'), indent=4)

    news_dataset_path = os.path.join(args.raw_dir, 'recsys/mind_train/news.json')
    save_path = os.path.join(args.raw_dir, 'recsys/mind_train/news_ml.json')
    news_dataset = json.load(open(news_dataset_path))
    new_news_dataset = []
    for news in news_dataset:
        new_news_dataset.append(id_2_news[news['id']])
    json.dump(new_news_dataset, open(save_path, 'w'), indent=4)

    save_path = os.path.join(args.raw_dir, 'generator/mind_train/news_ml.json')
    json.dump(new_news_dataset, open(save_path, 'w'), indent=4)

    imp_dataset_path = os.path.join(args.raw_dir, 'recsys/mind_test/behaviors.json')
    save_path = os.path.join(args.raw_dir, 'recsys/mind_test/behaviorsl_nl.json')
    imp_dataset = json.load(open(imp_dataset_path))
    new_imp_dataset = []
    for imp in imp_dataset:
        past_clicked_content = []
        for news_idx in imp['past_clicked']:
            past_clicked_content.append(id_2_news[news_idx])
        imp['past_clicked_content'] = past_clicked_content
        clicked_news_content = []
        for news_idx in imp['clicked_news']:
            clicked_news_content.append(id_2_news[news_idx])
        imp['clicked_news_content'] = clicked_news_content
        new_imp_dataset.append(imp)
    json.dump(new_imp_dataset, open(save_path, 'w'), indent=4)

    imp_dataset_path = os.path.join(args.raw_dir, 'generator/mind_test/behaviors.json')
    save_path = os.path.join(args.raw_dir, 'generator/mind_test/behaviorsl_nl.json')
    imp_dataset = json.load(open(imp_dataset_path))
    new_imp_dataset = []
    for imp in imp_dataset:
        past_clicked_content = []
        for news_idx in imp['past_clicked']:
            past_clicked_content.append(id_2_news[news_idx])
        imp['past_clicked_content'] = past_clicked_content
        clicked_news_content = []
        for news_idx in imp['clicked_news']:
            clicked_news_content.append(id_2_news[news_idx])
        imp['clicked_news_content'] = clicked_news_content
        new_imp_dataset.append(imp)
    json.dump(new_imp_dataset, open(save_path, 'w'), indent=4)

    news_dataset_path = os.path.join(args.raw_dir, 'recsys/mind_test/news.json')
    save_path = os.path.join(args.raw_dir, 'recsys/mind_test/news_ml.json')
    news_dataset = json.load(open(news_dataset_path))
    new_news_dataset = []
    for news in news_dataset:
        new_news_dataset.append(id_2_news[news['id']])
    json.dump(new_news_dataset, open(save_path, 'w'), indent=4)

    save_path = os.path.join(args.raw_dir, 'generator/mind_test/news_ml.json')
    json.dump(new_news_dataset, open(save_path, 'w'), indent=4)



