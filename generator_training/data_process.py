import json
import argparse

def my_summary_data_process(behaviors):
    inputs = []
    instructions = []
    outputs = []
    ori_inputs = []

    instruction = """You are a personalized text generator. First, I will provide you with a news list that includes both the [main news] and [topic-related news]. Second, I will provide you with user interests, including the [categories] and [topics] of news that the user is interested in.
Based on the input news list and user interests, you are required to generate a {personalized news summary} centered around the [main news]. 
"""
    for imp in behaviors:
        temp = 'News List:\n[\n'
        temp = temp + '{"ID": "Main News", "title": "' + imp['clicked_news_content'][0]['title'] + \
               '", "category": "' + imp['clicked_news_content'][0]['category'] + \
               '", "topics": "' + ', '.join(imp['clicked_news_content'][0]['topic']) + \
               '", "abstract": "' + imp['clicked_news_content'][0]['abstract'] + '\n'

        for n_idx, related_news in enumerate(imp['related_news_content']):
            temp = temp + '{"ID": "Topic-related News ' + str(n_idx + 1) + \
                   '", "title": "' + related_news['title'] + \
                   '", "category": "' + related_news['category'] + \
                   '", "topics": "' + ', '.join(related_news['topic']) + \
                   '", "abstract": "' + related_news['abstract'] + '"},\n'

        temp = temp[:-2] + '\n'
        temp = temp + ']\n'

        temp = temp + 'User Interest:\n'
        for up in imp['user_profile']:
            temp = temp + 'This user is interested in news about [' + up['category'] + '], especially [' + ', '.join(
                up['topic']) + '].\n'

        inputs.append(temp)

        temp = '{"title": "' + imp['personalized_news']['title'] + \
               '",\n"category": "' + imp['personalized_news']['category'] + \
               '",\n"topics": "' + imp['personalized_news']['topic'] + \
               '",\n"abstract": "' + imp['personalized_news']['abstract'] + '"}'

        outputs.append(temp)


        instructions.append(instruction)

        imp.pop('event_news', None)
        imp.pop('event_news_content', None)
        ori_inputs.append(imp)

    return inputs, outputs, instructions, ori_inputs

def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--training_dataset', type=str)
    parse.add_argument('--testing_dataset', type=str)
    parse.add_argument('--training_save_path', type=str)
    parse.add_argument('--testing_save_path', type=str)
    args = parse.parse_args()
    return args

args = my_parse()

training_dataset = json.load(open(args.training_dataset))
inputs, outputs, instructions, ori_inputs = my_summary_data_process(training_dataset)
final = []
for i in range(len(inputs)):
    final.append({
        'instruction': instructions[i],
        'input': inputs[i],
        'output': outputs[i]
    })
json.dump(final, open(args.training_save_path, 'w'), indent=4)

testing_dataset = json.load(open(args.testing_dataset))
inputs, outputs, instructions, ori_inputs = my_summary_data_process(testing_dataset)
final = []
for i in range(len(inputs)):
    final.append({
        'instruction': instructions[i],
        'input': inputs[i],
        'output': outputs[i]
    })
json.dump(final, open(args.testing_save_path, 'w'), indent=4)
