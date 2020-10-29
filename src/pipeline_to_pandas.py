import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup

class pipelineToPandas():
    '''
    Class that performs ETL from datalake json file to structured pandas dataframe.
    '''
    def __init__(self):
        self.base_path = '../data/'
        '''
        SELECT 
            id AS tweet_id,
            state,
            search_term_key,
            created_at AS tweet_date_created,
            text AS tweet_text,
            display_text_range AS tweet_text_range,
            source,
            user.`id` AS user_id,
            user.`created_at` AS user_date_created,
            user.`location` AS location,
            user.`description` AS description,
            user.`verified` AS user_verified
        FROM sql_temp_table
        WHERE
            lang = 'en'
        '''

    def clean_df(self, all=True):
        '''
        Clean df. This includes,
            dropping duplicate rows (based on tweet_id)
            eliminated mentions from tweet text
            format source from html to iPhone, Android, Web, Other
            format as datetime
        '''
        if all:
            # dropping duplicates
            self.df_all.drop_duplicates(subset='tweet_id', ignore_index=True, inplace=True)

            # Eliminating mentions, dropping 
            range_as_tuple = self.df_all['tweet_text_range'].apply(self._format_text_range) 
            self.df_all['tweet_text_wo_mentions'] = self._no_mentions_text(
                range_as_tuple,
                self.df_all['tweet_text']
            )
            self.df_all.drop(['tweet_text', 'tweet_text_range'], axis=1, inplace=True)  

            # formatting source
            self.df_all['source_text'] = self.df_all['source'].apply(lambda x: BeautifulSoup(x).find('a').getText())
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
            self.df_all['tweet_date_created'] = pd.to_datetime(self.df_all['tweet_date_created'])
            self.df_all['user_date_created'] = pd.to_datetime(self.df_all['user_date_created'])

    def _format_text_range(self, raw_range):
        '''
        Formats str tweet_range as tuple if present, otherwise leave as nan
        '''
        if type(raw_range) == str: 
            start, end = ''.join([char for char in raw_range if char not in ['[', ',', ']']]).split(' ')
            return (int(start), int(end))
        else:
            return raw_range

    def _no_mentions_text(self, tup_range_series, raw_tweet_text_series):
        wo_mentions = np.empty(raw_tweet_text_series.shape, dtype='U256')
        for idx, tup in enumerate(tup_range_series):
            if type(tup) == tuple:
                wo_mentions[idx] = raw_tweet_text_series[idx][tup[0]:tup[1]]
            else:
                wo_mentions[idx] = raw_tweet_text_series[idx]
        return wo_mentions
    
    def save_to_csv(self, csv_file_name, all=True):
        now_date = str(datetime.now()).split(' ')[0]
        file_path = f'{self.base_path}{now_date}_{csv_file_name}'
        if all:
            self.df_all.to_csv(file_path, index=False)
        else:
            self.pandas_df.to_csv(file_path, index=False)
        

if __name__ == "__main__":

    print('hi')
    pipeline = pipelineToPandas()
    pipeline.load_multiple_files()
    pipeline.clean_df()
    pipeline.save_to_csv('all_clean.csv')