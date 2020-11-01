# Python builtin libraries
from datetime import datetime
import re

# External libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

'''
# ETL steps

- Get data from Source:
    1. Info from this gist https://gist.github.com/tanaikech/f0f2d122e05bf5f971611258c22c110f
    2. For us:
        - Move 'download_full_ds.sh' to 'data' folder
        - Execute script

- load data from .jsonl file in Google drive
- Apply Transforms:

    1. Select only english tweets
    2. Select only tweets with user description (?)
    3. Drop unrelevant columns
    4. Apply Clustering to user descriptions and apply label (for hard-clustering) or latent features (soft)
    5. Apply Vader transform, apply compound score
    6. Apply pro-Trump/pro-Biden score based on hashtag dictionary, maybe to user desc text also
    7. Apply attach/non-attack classifier
    
    
- Load to usable form:

    1. Subset dataset to 4 subsets: proTrump/highInt, proTrump/lowInt, proBiden/highInt, proBiden/lowInt
    2. Save each as a jsonl file, format "1_1" for proTrump/highInt (if time, save to SQL/Lamda API)
    

- Script to run GPT-2 on each subset
    1. Could use [this medium post](https://medium.com/@ngwaifoong92/beginners-guide-to-retrain-gpt-2-117m-to-generate-custom-text-content-8bb5363d8b7f) for guidance
    2. Save generated texts in files with tweet_gen_ID for ref.
'''

class pipelineToPandas():
    '''
    Class that performs ETL from datalake json file to structured pandas dataframe.
    '''
    def __init__(self):
        self.base_path = '../data/'

    def load_json(self, fname, chunk=True, chunk_size = 10000):
        '''
        Loads json file to pandas df

        INPUT: fname <str>: File name
        '''
        self.chunk = chunk
        self.chunk_size = chunk_size

        if chunk:
            self.df_chunk_iter = pd.read_json(f'{self.base_path}{fname}', lines=True, chunksize=chunk_size)

        else:
            self.df_raw = pd.read_json(f'{self.base_path}{fname}', lines=True)
    
    def clean_df(self):
        '''
        Clean df. This includes,
            dropping duplicate rows (based on tweet_id)
            eliminated mentions from tweet text
            format source from html to iPhone, Android, Web, Other
            format as datetime
        '''

        cols_to_keep = ['id', 'full_text', 'source', 'entities', 'user', 'lang']

        if self.chunk:
            # select english tweets
            for chunk_id, chunk in enumerate(self.df_chunk_iter):
                # Subset columns
                chunk = chunk[cols_to_keep]

                # Select english tweets
                chunk = chunk[chunk['lang'] == 'en']

                # Drop duplicate tweets
                chunk.drop_duplicates(subset='id', ignore_index=True, inplace=True)

                # Grab user description
                chunk['user_desc'] = [chunk['user'][ind].get('description') for ind in range(len(chunk))]
                # chunk = chunk[chunk['user_desc'] != '']

                # Grab hashtags
                chunk['hashtags'] = [chunk['entities'][ind].get('hashtags') for ind in range(len(chunk))]

                # Return chunk

                # print update
                print(f'Cleaning chunks:\t{chunk_id+1} of {(self.chunk_size // len(chunk))+1} clean')

        else:
            # select english tweets
            self.df_all = self.df_all[self.df_all['lang'] == 'en']

            # dropping duplicates
            self.df_all.drop_duplicates(subset='id', ignore_index=True, inplace=True)

            # Eliminating mentions, dropping 
            range_as_tuple = self.df_all['display_text_range'].apply(self._format_text_range) 
            self.df_all['tweet_text_wo_mentions'] = self._no_mentions_text(
                range_as_tuple,
                self.df_all['full_text']
            )
            self.df_all.drop(['display_text_range'], axis=1, inplace=True)  

            # formatting source
            self.df_all['source_text'] = self.df_all['source'].apply(lambda x: self._get_atag_text(x))
            self.df_all.drop('source', axis=1, inplace=True)
            source_class_dict = {'Twitter for iPhone': 'iPhone', 'Twitter for Android': 'Android',
                        'Twitter Web App': 'Web', 'Twitter for iPad': 'iPad'}
            self.df_all['source_text'] = self.df_all['source_text'].replace(source_class_dict)
            self.df_all['source_text'].where(
                self.df_all['source_text'].apply(lambda x: x in source_class_dict.values()),
                'Other',
                inplace=True
            )

            # format datetime cols
            self.df_all['tweet_date_created'] = pd.to_datetime(self.df_all['created_at'])

    def _subset_tweets(self, df):
        '''
        Subsets columns, selects english, drops duplicates, grabs user_desc & hashtags for each tweet
        '''

        cols_to_keep = ['id', 'full_text', 'source', 'entities', 'user', 'lang']

        # Subset columns
        df = df[cols_to_keep]

        # Select english tweets
        df = df[df['lang'] == 'en']

        # Drop duplicate tweets
        df.drop_duplicates(subset='id', ignore_index=True, inplace=True)

        # Grab user description
        df['user_desc'] = [df['user'][ind].get('description') for ind in range(len(df))]

        # Grab hashtags
        df['hashtags'] = [df['entities'][ind].get('hashtags') for ind in range(len(df))]

        

    def _get_atag_text(self, tweet_text):
        a_tag = BeautifulSoup(tweet_text).find('a')
        if type(a_tag) == str:
            return a_tag.getText()

    def _format_text_range(self, raw_range):
        '''
        Formats str tweet_range as tuple if present, otherwise leave as nan
        '''
        if type(raw_range) == str: 
            start, end = ''.join([char for char in raw_range if char not in ['[', ',', ']']]).split(' ')
            return (int(start), int(end))
        else:
            return raw_range

    # Doesn't work yet 10/29

    def _no_mentions_text(self, tup_range_series, raw_tweet_text_series):
        wo_mentions = np.empty(raw_tweet_text_series.shape, dtype='U256')
        for idx, tup in enumerate(tup_range_series):
            if type(tup) == tuple:
                wo_mentions[idx] = raw_tweet_text_series[idx][tup[0]:tup[1]]
            else:
                wo_mentions[idx] = raw_tweet_text_series[idx]
        return wo_mentions
    
    # def save_to_csv(self, csv_file_name, all=True):
    #     now_date = str(datetime.now()).split(' ')[0]
    #     file_path = f'{self.base_path}{now_date}_{csv_file_name}'
    #     if all:
    #         self.df_all.to_csv(file_path, index=False)
    #     else:
    #         self.pandas_df.to_csv(file_path, index=False)
        

if __name__ == "__main__":
    pipeline = pipelineToPandas()
    pipeline.load_json('concatenated_abridged.jsonl')
    pipeline.clean_df()