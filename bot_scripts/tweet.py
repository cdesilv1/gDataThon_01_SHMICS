#!/usr/bin/env python
# gDataThon_01_SHMICS/tweet.py

import tweepy
import logging
from config import create_api
import time
import random
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_fnames_helper() -> list:
    # get fnames from helper
    with open("count_logger.json", "r") as fread:
        read_json = json.load(fread)
        fnames = list(read_json.keys())
    return fnames


def make_tweet(api, filenames: list = get_fnames_helper()):
    logger.info("Deciding on tweet type to post")

    # pop to cycle through files if one has been iterated through completely
    def select_tweet(filenames=filenames) -> str:
        files_to_select_from = list(range(len(filenames)))
        random.shuffle(files_to_select_from)

        while files_to_select_from:
            current_f = filenames[files_to_select_from.pop()]

            # reading next idx of selected file
            with open("count_logger.json", "r") as reading_json:
                read_json_ct_logger = json.load(reading_json)
                next_idx_current_file = read_json_ct_logger[current_f]
                next_idx_current_file = str(next_idx_current_file)

            # reading the content of the next idx of selected file

            found_new_tweet = False

            with open(current_f, "r") as fread:
                read_json = json.load(fread)
                try:

                    def _recursive_helper(
                        read_json,
                        next_idx_current_file,
                        recursive_iterator=0,
                        idx_max=5000,
                    ):
                        find_bw_re = r"\<\|startoftext\|\>(.*?)\<\|endoftext\|\>"
                        tweet_content = read_json[
                            str(next_idx_current_file + recursive_iterator)
                        ]
                        found_content = re.findall(find_bw_re, tweet_content)
                        if next_idx_current_file + recursive_iterator <= idx_max:
                            if found_content:
                                return (
                                    found_content[0],
                                    next_idx_current_file + recursive_iterator,
                                )
                            else:
                                return _recursive_helper(
                                    read_json=read_json,
                                    next_idx_current_file=next_idx_current_file,
                                    recursive_iterator=recursive_iterator + 1,
                                )
                        else:
                            return "Sorry, we're out of fresh tweets.", 5000

                    tweet_content, current_idx = _recursive_helper(
                        read_json=read_json, next_idx_current_file=next_idx_current_file
                    )
                    found_new_tweet = True
                except:
                    logger.info("file: {} is out of tweets".format(current_f))

            if found_new_tweet:
                # write the next idx of selected file
                read_json_ct_logger[current_f] = current_idx
                with open("count_logger.json", "w") as fwrite:
                    fwrite.write(json.dumps(read_json_ct_logger))

                return tweet_content

        else:
            tweet_content = "Sorry, we're out of fresh tweets."

        return tweet_content

    logger.info("Updating tweet")

    tweet_content = select_tweet()

    api.update_status(tweet_content)


def main():
    api = create_api()
    while True:
        make_tweet(api)
        sleep_time = random.randint(2400, 3600)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
