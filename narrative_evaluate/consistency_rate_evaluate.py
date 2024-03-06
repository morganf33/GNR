import os
import openai
import json
import argparse
import time
from pathlib import Path


class Generator:
    def __init__(self, api_key):
        openai.api_key = api_key

        self.prompt = {"role": "system",
                       "content": """Decide if the following summary is consistent with the corresponding article. Note that consistency means all information in the summary is supported by the article."""}

    def ask(self, article, summary):
        dialog = []
        dialog.append(self.prompt)
        dialog.append({"role": "user", "content": "Article:\n" + article + "Summary:\n" + summary + "Answer (ONLY Yes or No):\n"})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=dialog,
            stream=False,
            timeout=360
        )
        return response['choices'][0]['message']['content']

def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--output_dir', type=str)
    parse.add_argument('--output_file', type=str)
    parse.add_argument('--input_file', type=str)
    parse.add_argument('--api_key', type=str)
    args = parse.parse_args()
    return args

def main(args):
    raw_inputs = json.load(open(args.input_file))
    generator = Generator(args.api_key)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    now_time = time.strftime("%Y-%m-%d-%H:%M:%S")
    save_path_json = os.path.join(args.output_dir, 'outputs_' + now_time + '.json')

    inputs = []
    final_raw_inputs = []
    for imp in raw_inputs:
        temp_summary_article = {}
        idx = 0
        temp_summary_article['article'] = ""
        for text in imp['clicked_news_content']:
            temp_summary_article['article'] += 'Paragraph '+str(idx)+': '+text['title'] + ' ' + text['abstract']
            idx += 1
        for text in imp['related_news_content']:
            temp_summary_article['article'] += 'Paragraph '+str(idx)+': '+text['title'] + ' ' + text['abstract']
            idx += 1

        temp_summary_article['summary'] = imp['personalized_news']['title'] + ' ' + imp['personalized_news']['abstract']
        inputs.append(temp_summary_article)
        final_raw_inputs.append(imp)

    for idx, temp_summary_article in enumerate(inputs):
        response = generator.ask(temp_summary_article['article'], temp_summary_article['summary'])
        final_raw_inputs[idx]['consistency_score'] = response

    json.dump(final_raw_inputs, open(save_path_json, 'w'), indent=4)
    return final_raw_inputs

def process_score(answer):
    all_score = 0
    all_imp = 0
    for data in answer:
        if 'consistency_score' not in data.keys():
            continue
        if 'Yes' in data['consistency_score']:
            all_score += 1
            all_imp += 1
        elif 'No' in data['consistency_score']:
            all_score += 0
            all_imp += 1
        else:
            print('Wrong ' + data['consistency_score'])
    print('consistency_rate: ', all_score / all_imp)

if __name__ == "__main__":
    args = my_parse()
    answer = main(args)
    process_score(answer)


