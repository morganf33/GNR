import openai
import json
import time
import argparse
import re
import os
from pathlib import Path


now_time = time.strftime("%Y-%m-%d-%H:%M:%S")


class Generator:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.prompt = {"role": "system",
                       "content": """
Based on the given news information, summarize what topic(s) the news is related to. Each news article is related to 1-3 topics, and each topic should not exceed five words.
Note:
You can only respond in the following format:
This news is related to [topic_1], [topic_2], ...
You are not allowed to response any other words for any explanation or note. JUST GIVE ME the output."""}

        self.sample = {"role": "system",
                           "content": """
Now I'll give you an example. You should imitate it to complete subsequent formal tasks:
Input:
{"title": "'As long as it takes': Biden vows support for fire-ravaged Maui as search efforts continue",
"abstract": "President Joe Biden arrived in fire-ravaged Maui on Monday to witness the devastation left by an inferno more than a week ago and assess for himself a government response that some residents initially found lacking.",
"category": "politics"}
Output:
This news is related to [Joe Biden Visited Maui], [Maui wildfire].""",}

    def ask(self, question):
        dialog = []
        dialog.append(self.prompt)
        dialog.append(self.sample)
        dialog.append({"role": "user", "content": question + "Output:\n"})
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
    parse.add_argument('--input_file_list', type=list)
    parse.add_argument('--api_key', type=str)
    args = parse.parse_args()
    return args


def main(args):
    raw_inputs = []
    news_idx_set = set()
    for input_file in args.input_file_list:
        for news in json.load(open(input_file)):
            if news['id'] not in news_idx_set:
                news_idx_set.add(news['id'])
                raw_inputs.append(news)
            if 'related_news_content' in list(news.keys()):
                for r_news in news['related_news_content']:
                    if r_news['id'] not in news_idx_set:
                        news_idx_set.add(r_news['id'])
                        raw_inputs.append(r_news)

    generator = Generator(args.api_key)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_path_json = os.path.join(args.output_dir, 'outputs_' + now_time + '.json')

    inputs = []
    news_content = []

    for news in raw_inputs:
        temp = ""
        temp = temp + 'Input:\n'
        temp = temp + '{"title": "' + news['title'] + '",\n' + '"abstract": "' + news['abstract'] + '",\n' + \
               '"category": "' + news['category'] + '"}\n'
        inputs.append(temp)
        news_content.append(news)

    answer = []

    for idx, new_input in enumerate(inputs):
        temp_ans = news_content[idx]
        temp_ans['input'] = new_input

        response = generator.ask(new_input)

        temp_ans['output'] = response
        answer.append(temp_ans)

        json.dump(answer, open(save_path_json, 'w'), indent=4)

    return answer

def extract_placeholders(s):
    placeholders = []
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, s)
    for match in matches:
        placeholders.append(match)
    return placeholders

def process(args, answer):
    final_output = []
    for data in answer:
        data['topic'] = extract_placeholders(data['output'])
        data.pop('input', None)
        data.pop('output', None)
        final_output.append(data)
    json.dump(final_output, open(args.output_file, 'w'), indent=4)

if __name__ == "__main__":
    args = my_parse()
    answer = main(args)
    process(args, answer)

