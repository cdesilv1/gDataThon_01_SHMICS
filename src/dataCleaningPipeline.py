# Python builtin libraries
from datetime import datetime
import re
import os
import pickle

# External libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Other py files
from TextCleaner import TextCleaner

class tweetCleaner():
    '''
    Class that performs ETL from datalake json file to structured dataframe.
    '''
    def __init__(self):
        self.base_path = '../data/'

    def load_json(self, fname, chunk=True, chunk_size = 10000):
        '''
        Loads json file to pandas df, can load in by chunks to reduce memory requirement.

        INPUT: fname <str>: File name
        '''
        self.chunk = chunk
        self.chunk_size = chunk_size

        if chunk:
            self.num_chunks = (sum(1 for row in open(f'{self.base_path}{fname}', 'r')) // chunk_size) + 1
            self.df_chunk_iter = pd.read_json(f'{self.base_path}{fname}', lines=True, chunksize=chunk_size)

        else:
            self.df_raw = pd.read_json(f'{self.base_path}{fname}', lines=True)
    
    def clean_df(self, proc_file_dir):
        '''
        Remove unwanted fields from tweets, add engineered features, writes to new jsonl files.
        '''

        if self.chunk:
            # select english tweets
            for chunk_id, chunk in enumerate(self.df_chunk_iter):
                # Subset tweet cols
                chunk = self._subset_tweets(chunk)

                # Apply sentiment analysis
                chunk = self._sentiment_score_tweets(chunk)

                # Placeholder for proTrump/proBiden scorer
                chunk = self._partisan_score(chunk)

                # Filter tweets to subpopulations
                self._write_to_subpopulations(chunk, proc_file_dir)

                # print update
                print(f'Cleaning chunks:\t{chunk_id+1} of {self.num_chunks} clean')

        else:
            self.df_raw = self._subset_tweets(self.df_raw)
            
            self.df_raw = self._sentiment_score_tweets(self.df_raw)

            self.df_raw = self._partisan_score(self.df_raw)

            self._write_to_subpopulations(self.df_raw, proc_file_dir)

            # print update
            print(f'Entire file loaded and cleaned')

    def _write_to_subpopulations(self, df, proc_file_dir, sentiment_thresh=0.5):
        '''
        Filter DF by partisanship and sentiment: (populations subject to change)
            - Population 1 (proTrump): proTrump/positiveSent & proBiden/negativeSent
            - Population 2 (proBiden): proTrump/negativeSent & proBiden/positiveSent
        '''
        # make masks
        proTrump_mask = df['partisan_score'] == 1
        posSent_mask = df['vader_sentiment'] > sentiment_thresh

        proBiden_mask = df['partisan_score'] == 0
        negSent_mask = df['vader_sentiment'] < -sentiment_thresh

        # Population combos
        mask_dict = {
            'proTrump': (proTrump_mask & posSent_mask) | (proBiden_mask & negSent_mask),
            'proBiden': (proTrump_mask & negSent_mask) | (proBiden_mask & posSent_mask),
        }

        # Write to jsonl files
        for population, mask in mask_dict.items():
            fname = f'{population}_sentiment_thresh_{sentiment_thresh}.jsonl'
            df_copy = df[mask]
            with open(f'{proc_file_dir}/{fname}', 'a') as f:
                f.writelines(df_copy.to_json(orient='records', lines=True))
                f.writelines('\n')

    def _partisan_score(self, df):
        '''
        Apply algoritm to score tweet partisanship.
        '''
        self._load_partisan_models()



        # df['biden_proba'] = 

        return model_trump.predict_proba(vec_trump)[0][1],model_biden.predict_proba(vec_biden)[0][1]


        # Placeholder for now - replace with ML algo later (11/1, 3:20pm MDT)
        df['partisan_score'] = np.random.randint(0,1, size=(len(df),1))
        return df

    def _vectorize_tweet_text(self, df):
        '''
        Vectorize tweet text for both 
        '''
        # grab tweet text, and clean it
        cleaner = TextCleaner()
        df['clean_tweet_text'] = df['full_text'].apply(lambda x: cleaner.clean_tweets(x,6))
        
        # load stopwords
        stop_words = pd.read_csv(f'../models/stop_words.csv',header=None)[0].to_list()

        # Collect trump/biden vocab tweets
        for tweet in df['clean_tweet_text']:
            tweet = ' '.join([i.lower() for i in tweet.split() if i not in stop_words])
            tweet =  re.sub(r'[^\w\s]','',tweet)
            
            df['trump_text'] = ' '.join([i for i in tweet.split() if i in self.trump_vectorizer.vocabulary_.keys()])
            df['biden_text'] = ' '.join([i for i in tweet.split() if i in self.biden_vectorizer.vocabulary_.keys()])

        breakpoint()

        vec_trump=vectorizer_trump.transform([s_trump])
        vec_biden=vectorizer_biden.transform([s_biden])
        return vec_biden, vec_trump
    
    def _load_partisan_models(self, classifier=MultinomialNB, training_date='20_11_05'):
        '''
        Load partisan scoring models.
        '''
        self.biden_model = pickle.load(open(f'../models/NB_biden_{training_date}.sav', 'rb'))
        self.trump_model = pickle.load(open(f'../models/NB_trump_{training_date}.sav', 'rb'))

        self.biden_vectorizer = pickle.load(open(f'../models/vectorizer_biden_{training_date}.sav', 'rb'))
        self.trump_vectorizer = pickle.load(open(f'../models/vectorizer_trump_{training_date}.sav', 'rb'))

    def _sentiment_score_tweets(self, df):
        '''
        Apply VADER algo to tweet text, returns only compount score
        '''
        analyzer = SentimentIntensityAnalyzer()
        df['vader_sentiment'] = df['full_text'].apply(lambda x: (analyzer.polarity_scores(x)).get('compound'))
        return df

    def _subset_tweets(self, df):
        '''
        Subsets columns, selects english, drops duplicates, grabs user_desc & hashtags for each tweet
        '''

        cols_to_keep = ['id', 'full_text', 'entities', 'user', 'lang']

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

        # Drop unwanted fields
        df.drop(['user', 'lang', 'entities'], axis=1, inplace=True)

        return df

    
        

if __name__ == "__main__":
    pipeline = tweetCleaner()
    pipeline.load_json('concatenated_abridged.jsonl')
    # pipeline.clean_df('../data/proc_jsons')

    pipeline._load_partisan_models()

    for chunk_id, chunk in enumerate(pipeline.df_chunk_iter):
        if chunk_id == 0:
            vec_biden, vec_trump = pipeline._vectorize_tweet_text(chunk)
