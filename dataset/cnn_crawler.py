from bs4 import BeautifulSoup
import requests
import os
import re
import json
import time
from datetime import date, datetime
import xml.dom.minidom
from pathlib import Path

def related_download(news_url):
    check_1 = re.match('https://www.cnn.com/', news_url)
    check_2 = re.match('https://edition.cnn.com/', news_url)
    if check_1 is None and check_2 is None:
        return None
    resp = requests.get(news_url, timeout=3)
    resp.encoding='utf-8'
    html = resp.text
    return html


related_idx = 0
related_url_2_news = {}
def related_content_download(news_url):
    global related_idx
    global related_url_2_news
    if news_url in list(related_url_2_news.keys()):
        return related_url_2_news[news_url]
    html = related_download(news_url)
    if html is None:
        return None
    soup = BeautifulSoup(html, 'html.parser')
    try:
        title = json.loads(soup.find('script', {'type': 'application/ld+json'}).get_text())["headline"]
        abstract = json.loads(soup.find('script', {'type': 'application/ld+json'}).get_text())["description"]
    except BaseException:
        return None
    news_content = {'id': 'related_' + str(related_idx+1), 'title': title, 'category': topic_name, 'abstract': abstract}
    related_idx += 1
    related_url_2_news[news_url] = news_content
    return news_content


def related_news_crawler(url):
    html = related_download(url)
    soup = BeautifulSoup(html, 'html.parser')
    paragraph_list = soup.find_all('p', {'class': 'paragraph inline-placeholder'})

    related_content_list = []
    for para in paragraph_list:
        link = para.find('a', {'target': '_blank'})
        if link is None:
            continue
        link = link['href']
        check_1 = re.match('https://www.cnn.com/', link)
        check_2 = re.match('https://edition.cnn.com/', link)
        if check_1 is None and check_2 is None:
            continue
        related_content = related_content_download(link)
        if related_content is None:
            continue
        related_content_list.append(related_content)
        if len(related_content_list) >= 3:
            break
    return related_content_list


topic_name = 'politics'

save_path = './generator/extract_news_with_related_news.json'

file_list = os.listdir('./extract_news')
file_list.sort(reverse=False)

news_source = set()
collect_news = []
for raw_file_name in file_list:
    if raw_file_name == '.DS_Store':
        continue
    file_name = os.path.join('./extract_news', raw_file_name + '/edition.cnn.com/' + topic_name)

    if not os.path.isfile(file_name):
        file_name = os.path.join('./extract_news', raw_file_name + '/edition.cnn.com/' + topic_name + '/index.html')

    if not os.path.isfile(file_name):
        os.path.join('./extract_news', raw_file_name + '/www.cnn.com/' + topic_name)

    html = open(file_name)
    raw_txt = ""
    for txt in html.readlines():
        if '{"articleList":' in txt:
            raw_txt = txt.split('{"articleList":')[1]
            raw_txt = raw_txt.split('registryURL')[0]
            break
    if len(raw_txt) <= 0:
        continue

    start_idx = raw_txt.find('{')
    end_idx = raw_txt.find('}')
    while len(raw_txt[start_idx:end_idx + 1]) > 0:
        temp_news = json.loads(raw_txt[start_idx:end_idx+1])
        if temp_news['uri'] in news_source:
            continue
        news_source.add(temp_news['uri'])
        related_news_content_list = related_news_crawler('https://www.cnn.com' + temp_news['uri'])
        collect_news.append({
            'id': 'extract_' + str(len(collect_news) + 1),
            'title': temp_news['headline'],
            'category': topic_name,
            'abstract': temp_news['description'],
            'related_news_content': related_news_content_list,
        })
        if end_idx+1 >= len(raw_txt):
            break
        raw_txt = raw_txt[end_idx+1:]
        start_idx = raw_txt.find('{')
        end_idx = raw_txt.find('}')
json.dump(collect_news, open(save_path, 'w'), indent=4)

