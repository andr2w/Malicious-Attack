import os
import json
import pandas as pd

no_dirs = ['all-rnr-annotated-threads/charliehebdo-all-rnr-threads/non-rumours']



data = []
for file in os.listdir('all-rnr-annotated-threads/charliehebdo-all-rnr-threads/non-rumours'):
    if file != '.DS_Store':
        source_tweets_path = f'all-rnr-annotated-threads/charliehebdo-all-rnr-threads/non-rumours/{file}/source-tweets'
        for f in os.listdir(source_tweets_path):
            file_path = source_tweets_path + '/' + f
            j = json.load(open(file_path, "rb"))
            source = j['text']
            data.append([source, 'non-rumors'])


for file in os.listdir('all-rnr-annotated-threads/charliehebdo-all-rnr-threads/rumours'):
    if file != '.DS_Store':
        source_tweets_path = f'all-rnr-annotated-threads/charliehebdo-all-rnr-threads/rumours/{file}/source-tweets'
        for f in os.listdir(source_tweets_path):
            file_path = source_tweets_path + '/' + f
            j = json.load(open(file_path, "rb"))
            source = j['text']
            data.append([source, 'rumors'])



for file in os.listdir('all-rnr-annotated-threads/ebola-essien-all-rnr-threads/non-rumours'):
    if file != '.DS_Store':
        source_tweets_path = f'all-rnr-annotated-threads/ebola-essien-all-rnr-threads/non-rumours/{file}/source-tweets'
        for f in os.listdir(source_tweets_path):
            file_path = source_tweets_path + '/' + f
            j = json.load(open(file_path, "rb"))
            source = j['text']
            data.append([source, 'non-rumors'])


for file in os.listdir('all-rnr-annotated-threads/ebola-essien-all-rnr-threads/rumours'):
    if file != '.DS_Store':
        source_tweets_path = f'all-rnr-annotated-threads/ebola-essien-all-rnr-threads/rumours/{file}/source-tweets'
        for f in os.listdir(source_tweets_path):
            file_path = source_tweets_path + '/' + f
            j = json.load(open(file_path, "rb"))
            source = j['text']
            data.append([source, 'rumors'])



for file in os.listdir('all-rnr-annotated-threads/ferguson-all-rnr-threads/non-rumours'):
    if file != '.DS_Store':
        source_tweets_path = f'all-rnr-annotated-threads/ferguson-all-rnr-threads/non-rumours/{file}/source-tweets'
        for f in os.listdir(source_tweets_path):
            file_path = source_tweets_path + '/' + f
            j = json.load(open(file_path, "rb"))
            source = j['text']
            data.append([source, 'non-rumors'])


for file in os.listdir('all-rnr-annotated-threads/ferguson-all-rnr-threads/rumours'):
    if file != '.DS_Store':
        source_tweets_path = f'all-rnr-annotated-threads/ferguson-all-rnr-threads/rumours/{file}/source-tweets'
        for f in os.listdir(source_tweets_path):
            file_path = source_tweets_path + '/' + f
            j = json.load(open(file_path, "rb"))
            source = j['text']
            data.append([source, 'rumors'])





df = pd.DataFrame(data, columns=['Text', 'Label'])
df.to_csv('data.csv')
