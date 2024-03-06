import openai
import json
import time
import argparse
import re

now_time = time.strftime("%Y-%m-%d-%H:%M:%S")


class Generator:
    def __init__(self, api_key):
        openai.api_key = api_key

        self.prompt = {"role": "system",
                       "content": """You are a personalized text generator. First, I will provide you with a news list that includes both the [main news] and [topic-related news]. Second, I will provide you with [user interest], including the [categories] and [topics] of news that the user is interested in.
Based on the input news list and user interests, you are required to generate a {personalized news summary} centered around the [main news]. The COHERENCE and FLUENCY of the {personalized news summary} are crucial! Therefore, your task is not simply to string the news together! Therefore, you can infer relationships between the news and selectively include only a few [topic-related news] that are relevant to [main news] in the {personalized news summary}.
You should highlight content in {personalized news summary} that is relevant to the [user interest] and try to ignore irrelevant content. The personality is very important!!!
Note:
You can only output the personalized news summary (only one) in the json format. And the personalized news summary should be less than 100 words!! (This is very Important!!)
You are not allowed to response any other words for any explanation or note. JUST GIVE ME JSON-FORMAT output.
"""}
        self.sample = {"role": "system",
                           "content": """Here are some examples:
News List:
[
{"ID": "Main News", "title": "Trump announces he will nominate deputy energy secretary to replace Rick Perry in top post", "category": "politics", "topics": "Trump's nomination, Dan Brouillette, Rick Perry replacement", "abstract": "President Donald Trump announced Friday that he plans to nominate Deputy Energy Secretary Dan Brouillette to replace Secretary of Energy Rick Perry."},
{"ID": "Topic-related News 1", "title": "Senate Confirms Dan Brouillette to Lead Energy Department", "category": "politics", "topics": "Dan Brouillette, Energy Department", "abstract": "The U.S. Senate voted 70-15 to confirm the nomination of Dan Brouillette as energy secretary."},
{"ID": "Topic-related News 2", "title": "Rick Perry informs Trump of his plans to resign later this year as scrutiny over Ukraine heats up", "category": "politics", "topics": "Rick Perry Resignation, Ukraine Scandal", "abstract": "Energy Secretary Rick Perry said on Thursday that he plans to leave his post later this year after he informed President Donald Trump of his intention to resign."},
{"ID": "Topic-related News 3", "title": "Subpoena for Rick Perry in House impeachment Inquiry", "category": "politics", "topics": "Rick Perry subpoena, Trump impeachment, Ukraine", "abstract": "House Democrats on Thursday issued a subpoena to Energy Secretary Rick Perry for documents related to the Trump administration’s contacts with Ukraine as part of the ongoing House impeachment inquiry."},
{"ID": "Topic-related News 4", "title": "Trump praises Rick Perry and confirms his departure at rally", "category": "politics", "topics": "Donald Trump, Rick Perry's Departure", "abstract": "President Trump recognized Energy Secretary Rick Perry, a Texas native, during his Dallas rally tonight, telling the crowd that Perry would be leaving at the end of the year."},
]
User Interest:
This user is interested in news about [politics], especially [Trump's Ukraine call, Trump's legal team, Rick Perry, US natural gas].
This user is interested in news about [us], especially [Trump attends Kushner's anniversary].
Personalized News Summary (Less than 100 words):
{
"title": "Trump Nominates Brouillette as Energy Secretary to Replace Resigning Perry",
"category": "politics",
"topics": "Trump's nomination, Rick Perry, Trump's Ukraine call",
"abstract": "President Trump plans to nominate Dan Brouillette to replace Rick Perry as Secretary of Energy. Previously, Perry announced his intention to resign later this year amidst scrutiny over Ukraine. Afterwards, Trump confirmed his departure and praised his work. Perry's successor is Deputy Energy Secretary Dan Brouillette, who was elected with a 70-15 vote.",
}

News List:
[
{"ID": "Main News", "title": "Federal Prosecutors Probe Giuliani's Links to Ukrainian Energy Projects", "category": "politics", "topics": "Rudy Giuliani, Ukrainian energy projects, Federal prosecutors", "abstract": "Federal prosecutors in New York are investigating whether Rudy Giuliani stood to personally profit from a Ukrainian natural-gas business pushed by two associates who also aided his efforts there to launch investigations that could benefit President Trump, people familiar with the matter said.
{"ID": "Topic-related News 1", "title": "Bannon: Trump 'is going to have to rethink his legal team'", "category": "politics", "topics": "Stephen Bannon's suggestion, Rudy Giuliani's removal", "abstract": "President Trump's former top aide Stephen Bannon says that his old boss should consider shaking up his legal team, and in particular appeared to call for the removal of Rudy Giuliani.In an interview airing Sunday on AM 970 "The Answer," Bannon told host John Catsimatidis that Giuliani had gotten "over his skis" in his contacts with Ukrainian government officials urging President Volodymyr Zelensky's administration to open a criminal..."},
{"ID": "Topic-related News 2", "title": "Giuliani's name mentioned several times in transcripts", "category": "politics", "topics": "Rudy Giuliani, Ukraine controversy", "abstract": "CNN's Anderson Cooper examines the role of President Donald Trump's personal attorney Rudy Giuliani in Ukraine controversy."},
{"ID": "Topic-related News 3", "title": "Giuliani associates Fruman and Parnas plead not guilty to campaign finance charges", "category": "politics", "topics": "Giuliani associates plead not guilty, campaign finance charges", "abstract": "Two men who have served as Rudy Giuliani’s conduit to Ukraine pleaded not guilty in Manhattan federal court Wednesday to charges that they funneled foreign money to US campaign coffers, with one of their lawyers suggesting executive privilege could apply to material in the case."}
]
User Interest:
This user is interested in news about [politics], especially [Impeachment Inquiry, Republican Lawmakers, Devin Nunes, Ukraine whistleblower, Donald Trump's fake tweet, G-7 summit, Trump National Doral, political corruption].
This user is interested in news about [2020 US Presidential Election], especially [Trump's Texas campaign, 2020 US Presidential Election, Joe Biden, Hillary Clinton, Trump's Doral resort, Security costs, Investigation].
Personalized News Summary (Less than 100 words):
{
"title": "Giuliani Under Federal Investigation for Ukrainian Energy Links", 
"category": "politics", 
"topic": "Rudy Giuliani, Ukrainian energy projects, Impeachment Inquiry", 
"abstract": "Federal prosecutors are investigating if Rudy Giuliani could personally benefit from a Ukrainian natural-gas business. His efforts in Ukraine could advantage President Trump. Giuliani's conduct in Ukraine has been under scrutiny, even leading to calls for his removal from Trump's legal team. He has been named multiple times in impeachment inquiry transcripts, highlighting his contested role."
}
"""}

    def ask(self, question):
        dialog = []
        dialog.append(self.prompt)
        dialog.append(self.sample)
        dialog.append({"role": "user", "content": question + "Personalized News Summary (Less than 100 words):\n"})
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
    save_path_json = args.output_dir + 'outputs_' + now_time + '.json'

    inputs = []
    final_raw_inputs = []

    for idx, imp in enumerate(raw_inputs):
        temp = 'News List:\n[\n'
        temp = temp + '{"ID": "Main News", "title": "' + imp['clicked_news_content'][0]['title'] + \
               '", "category": "' + imp['clicked_news_content'][0]['category'] + \
               '", "topics": "' + ', '.join(imp['clicked_news_content'][0]['topic']) + \
               '", "abstract": "' + imp['clicked_news_content'][0]['abstract'] + '\n'

        # temp = temp + '\nRelated News List:\n[\n'
        for idx, related_news in enumerate(imp['related_news_content']):
            temp = temp + '{"ID": "Topic-related News ' + str(idx + 1) + \
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


def process(args, answer):
    final_output = []
    for data in answer:
        if data['output'][-1] == '"':
            data['output'] = data['output'][:-1] + '"}'
        elif data['output'][-1] != '}':
            data['output'] = data['output'] + '"}'
        data['personalized_news'] = extract_info(data['output'])
        if data['personalized_news'] is None or len(data['personalized_news']) == 0:
            continue
        data.pop('input', None)
        data.pop('output', None)
        final_output.append(data)
    json.dump(final_output, open(args.output_file, 'w'), indent=4)


if __name__ == "__main__":
    args = my_parse()
    answer = main(args)
    process(args, answer)
