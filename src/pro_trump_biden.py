import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
# stop_words = pd.read_csv('stop_words.csv',header=None)[0].to_list() 
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def predict_(s):
    s=s.split('http')[0]
    s=' '.join([i.lower() for i in s.split() if i not in stop_words])
    s= re.sub(r'[^\w\s]','',s)
    s_trump=' '.join([i for i in s.split() if i in vectorizer_trump.vocabulary_.keys()])
    s_biden=' '.join([i for i in s.split() if i in vectorizer_biden.vocabulary_.keys()])

    vec_trump=vectorizer_trump.transform([s_trump])
    vec_biden=vectorizer_biden.transform([s_biden])
    return model_trump.predict_proba(vec_trump)[0][1],model_biden.predict_proba(vec_biden)[0][1]


def result(s,thresh_trump=0.5,thresh_biden=0.7):
    j,k=predict_(s)
    if j>=thresh_trump and k<=thresh_biden:
        return 'trump'
    elif j<=thresh_trump and k>=thresh_biden:
        return 'biden'
    else:
        return 'neither'

if __name__ == "__main__":
    vectorizer_trump = TfidfVectorizer(max_features=5000)
    vectorizer_biden = TfidfVectorizer(max_features=5000)


    X_biden=pd.read_csv('biden_data.csv')['0']
    X_trump=pd.read_csv('trump_data.csv')['0']

    vectorizer_trump.fit(X_trump)
    vectorizer_biden.fit(X_biden)

    with open('../models/vectorizer_biden_20_11_05.sav', 'wb') as f:
        pickle.dump(vectorizer_biden,f)


    with open('../models/vectorizer_trump_20_11_05.sav', 'wb') as f:
        pickle.dump(vectorizer_trump,f)