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
                       "content": """You are asked to describe user interest based on his/her browsed news list. User interest includes the news [categories] and news [topics] (under each [category]) that users are interested in.
You can only respond in the following format:
According to [browsed news ID_1, browsed news ID_2, ...], this user is interested in news about [news category_1], especially [news topic_1 related to news category_1, ...].
According to [browsed news ID_3, browsed news ID_4, ...], this user is interested in news about [news category_2], especially [news topic_2 related to news category_2, ...].
Note:
The number of news [category] that the user is interested in should not exceed 3! And the number of news [topic] under each [category] should not exceed 5!!
You are not allowed to response any other words for any explanation or note. JUST GIVE ME the output."""}

        self.sample = {"role": "system",
                           "content": """Now I'll give you an example. You should imitate it to complete subsequent formal tasks:
Input:
Browsed News List:
[
{"ID": "P_1", "title": "Texas police officer shoots woman to death inside her home", "category": "us", "topic": "shooting incidents, police-related cases, Texas"},
{"ID": "P_2", "title": "Missing Texas boy, 6, and his dog, found in cornfield by drone with thermal camera", "category": "us", "topic": "missing cases, missing boy, Texas"},
{"ID": "P_3", "title": "Politico: Trump lures GOP senators on impeachment with cold cash", "category": "politics", "topic": "Trump impeachment, GOP, cold cash"},
{"ID": "P_4", "title": "Elderly getting scammed by their own families, and AARP is out to stop it", "category": "us", "topic": "Elderly fraud, AARP"},
{"ID": "P_5", "title": "Trump administration to pay $846G to California over failed bid to add citizenship question to census", "category": "politics", "topic": "Trump administratio's citizenship question, California"},
{"ID": "P_6", "title": "What We've Learned From Impeachment Inquiry", "category": "politics", "topic": "Trump impeachment inquery"},
{"ID": "P_7", "title": "Newly released Trump tax documents show major inconsistencies", "category": "politics", "topic": "Trump tax documents"}
]
Output:
According to [P_1, P_2, P_4], this user is interested in news about [criminal], especially [criminal cases in Texas, shooting incidents, missing cases, Elderly fraud].
According to [P_3, P_5, P_6, P_7], this user is interested in news about [politics], especially [Trump impeachment, Trump administratio's citizenship questio, Trump tax documents]."""}

    def ask(self, question):
        dialog = []
        dialog.append(self.prompt)
        dialog.append(self.sample)
        dialog.append({"role": "user", "content": "Input:\nBrowsed News List: [" + question + "]\n Output:\n"})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=dialog,
            stream=False,
            timeout=360
        )
        return response['choices'][0]['message']['content']

def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--output_dir', type=str, default=None)
    parse.add_argument('--output_file', type=str, default=None)
    parse.add_argument('--input_file', type=str, default=None)
    parse.add_argument('--api_key', type=str, default=None)
    args = parse.parse_args()
    return args

def main(args):
    raw_inputs = json.load(open(args.input_file))
    generator = Generator(args.api_key)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_path_json = os.path.join(args.output_dir, 'outputs_' + now_time + '.json')

    inputs = []
    final_raw_inputs = []
    for idx, imp in enumerate(raw_inputs):
        temp = "\n"
        for past_news in imp['past_clicked_content']:
            temp = temp + '{"ID": ' + past_news['id'] + ', "title": "' + past_news['title'] + ', "category": ' + past_news['category'] + ', "topic": ' + ','.join(past_news['topic']) + '"},\n'
        temp = temp[:-1] + "\n"
        inputs.append(temp)
        final_raw_inputs.append(imp)

    answer = []
    for idx, new_input in enumerate(inputs):
        temp_ans = final_raw_inputs[idx]
        temp_ans['input'] = new_input
        response = generator.ask(new_input)
        temp_ans['output'] = response
        answer.append(temp_ans)
        json.dump(answer, open(save_path_json, 'w'), indent=4)
    return answer


def extract_info_new(s):
    pattern = r'According to.*?(?=According to|$)'
    sentence_list = re.findall(pattern, s, flags=re.DOTALL)
    result = []
    for sentence in sentence_list:
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, sentence)
        if len(matches) > 3:
            browsed_news_ids = matches[0]
            category = matches[1]
            topic = matches[2:]
        elif len(matches) == 3:
            browsed_news_ids = matches[0]
            category = matches[1]
            topic = matches[2].split(', ')
        else:
            return None
        result.append((browsed_news_ids, category, topic))
    return result


def process(args, answer):
    final_output = []
    for data in answer:
        user_profile = extract_info_new(data['output'])
        if user_profile is None or len(user_profile) == 0:
            continue
        final_user_profile = []
        for t in user_profile:
            final_user_profile.append({
                'news_id': t[0],
                'category': t[1],
                'topic': t[2]
            })
        data.pop('input', None)
        data.pop('output', None)
        data['user_profile'] = final_user_profile
        final_output.append(data)

        json.dump(final_output, open(args.output_file, 'w'), indent=4)

if __name__ == "__main__":
    args = my_parse()
    answer = main(args)
    process(args, answer)

