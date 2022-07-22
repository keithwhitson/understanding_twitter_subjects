#! .venv/bin/python

from unittest.util import _MAX_LENGTH
import pandas as pd
import snscrape.modules.twitter as sntwitter
import re

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

def clean_twitter_handle(handle):
    '''Removes the at symbol from Twitter Handle'''
    handle = handle.strip()
    handle = handle.replace('@','')
    handle = handle.strip()
    return handle

def get_twittter_handles():
    '''Get the list of Twitter Black Influencer Handles'''
    twitterHandles_df = pd.read_parquet('blacktwitter_influencers.parquet',columns=['TwitterHandle'])
    twitterHandles = twitterHandles_df['TwitterHandle'].values
    twitterHandles = [clean_twitter_handle(handle) for handle in twitterHandles]
    return twitterHandles

def cleaned_tweet(tweet) -> str:
    '''Removes whitespace and generally cleans tweet'''
    removed_link_tweet = re.sub(r"http\S+", "", tweet)
    removed_mention_tweet = re.sub("@[A-Za-z0-9]+", "", removed_link_tweet)
    removed_non_alpha = re.sub(r'[\W_]+', ' ', removed_mention_tweet)
    removed_rt = removed_non_alpha.replace('RT ', '')
    single_spaced_tweet = " ".join(removed_rt.split())
    lowercased_tweet = single_spaced_tweet.lower()
    return lowercased_tweet

def get_last_two_hundred_tweets():
    '''Gets the last two hundred tweets'''
    twitterHandles = get_twittter_handles()
    combined_text = ''
    for handle in twitterHandles:
        for ix, tweet in enumerate(sntwitter.TwitterUserScraper(handle).get_items()):
            tweet_text = tweet.content
            tweet_text = cleaned_tweet(tweet_text)
            combined_text = combined_text + ' ' + tweet_text
            if ix > 100:
                break
        break
    return combined_text

def main():
    '''The main function'''
    # Transform input tokens
    #inputs = tokenizer(get_last_two_hundred_tweets(), return_tensors="pt",model_max_length=12)
    #outputs = model(**inputs)
    #print(outputs)
    tokenized_text = tokenizer.encode(get_last_two_hundred_tweets(), return_tensors="pt").to(device)

    # summmarize
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=100,
                                        early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print ("\n\nSummarized text: \n",output)

if __name__ == "__main__":
    main()