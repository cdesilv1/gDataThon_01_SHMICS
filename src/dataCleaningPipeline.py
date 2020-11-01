# Python builtin libraries
from datetime import datetime
import re

# External libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

class tweetCleaner():
    '''
    Class that performs ETL from datalake json file to structured dataframe.
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

        '''

        if self.chunk:
            # select english tweets
            for chunk_id, chunk in enumerate(self.df_chunk_iter):
                # Subset tweet cols
                chunk = self._subset_tweets(chunk)

                # Apply sentiment analysis
                chunk = self._sentiment_score_tweets(chunk)

                # Placeholder for attack/non-attack classifier
                chunk = self._id_attack_tweets(chunk)


                # print update
                print(f'Cleaning chunks:\t{chunk_id+1} of {(self.chunk_size // len(chunk))+1} clean')

        else:
            self.df_raw = self._subset_tweets(self.df_raw)
            
            self.df_raw = self._sentiment_score_tweets(self.df_raw)

            self.df_raw = self._id_attack_tweets(self.df_raw)


    def _id_attack_tweets(self, df):
        '''
        Apply algorithm to classify tweets as attack/non-attack
        '''
        # Placeholder for now - replace with ML algo later (11/1, 3:20pm MDT)
        df['is_attack'] = np.random.randint(0,1,size=(len(df),1))
        return df

    def _sentiment_score_tweets(self, df):
        '''
        Apply VADER algo to tweet text, returns only compount score
        '''
        analyzer = SentimentIntensityAnalyzer()
        df['vader_sentiment'] = df.apply(lambda x: (analyzer.polarity_scores(x)).get('compound'))
        return df

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

        return df
        

if __name__ == "__main__":
    pipeline = pipelineToPandas()
    pipeline.load_json('concatenated_abridged.jsonl')
    pipeline.clean_df()