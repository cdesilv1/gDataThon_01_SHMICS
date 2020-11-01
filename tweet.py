#!/usr/bin/env python
# gDataThon_01_SHMICS/tweet.py

import tweepy
import logging
from config import create_api
import time
import random
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def make_tweet(api, filenames:list):
    logger.info("Deciding on tweet type to post")
    file_to_select_from = random.choice(filenames)

    # reading next idx of selected file
    with open('count_logger.json', 'r') as reading_json:
        read_json_ct_logger = json.load(reading_json)
        next_idx_current_file = read_json_ct_logger[file_to_select_from]
    
    # reading the content of the next idx of selected file
    with open(file_to_select_from, 'r') as fread:
        read_json = json.load(fread)
        try:
            tweet_content = read_json[next_idx_current_file]
        except:
            logger.info("idx: {} failed on file: {}".format(next_idx_current_file, file_to_select_from))
            tweet_content = "Sorry, we're out of fresh tweets."

    logger.info("Updating tweet")
    
    api.update_status(tweet_content)

    # write the next idx of selected file
    read_json_ct_logger[file_to_select_from] += 1
    with open('count_logger.json', 'w') as fwrite:
        fwrite(json.dumps(read_json_ct_logger))

def main():
    api = create_api()
    while True:
        make_tweet(api, ['FILENAMES HERE'])
        time.sleep(3600)

if __name__ == "__main__":
    main()
