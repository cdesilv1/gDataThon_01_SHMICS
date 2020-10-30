#%%

import re
from sklearn.feature_extraction import stop_words
stopwords = stop_words.ENGLISH_STOP_WORDS
nltk.download('vader_lexicon')

class TextCleaner():
    '''
    This class instantiates an object with attributes of text preprocessing dictionaries and 
    a method for applying this to a list of text. 
    '''
    def __init__(self):
        '''
        Removed groups: 
            r"[!?$%()*+,-./:;<=>\^_`{|}~]"
        '''
        self.re_substitution_groups = [r'^RT', r'^rt', r'http\S+', r'&amp; ', r'^[@#]\w+']
        self.text_abbrevs = { 'lol': 'laughing out loud', 'bfn': 'bye for now', 'cuz': 'because',
                            'afk': 'away from keyboard', 'nvm': 'never mind', 'iirc': 'if i recall correctly',
                            'ttyl': 'talk to you later', 'imho': 'in my honest opinion', 'brb': 'be right back',
                            "fyi": "for your information" }
        self.grammar_abbrevs = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                             "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                             "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                             "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                             "mustn't":"must not", "'s":"s"}


    def clean_tweets(self, df_tweet_text, last_clean_step=6):
        '''
        INPUT: df_tweet_text <string>
        This function will clean the text of tweets, with ability to very the last step of cleaning.
        order:
        1. lowercase
        2. change txt abbreviations
        3. change grammar abbreviation
        4. remove punctuation
        5. remove special (utf-8) characters
        6. remove stop words
        Run on one tweet at a time, for example:
        cleaner = TextCleaner()
        df['clean_tweets'] = df['full_text'].apply(lambda x: cleaner.clean_tweets(x, 5))
        '''
        df_tweet_text_sw = str(df_tweet_text)

        if last_clean_step == 0:
            clean_text = df_tweet_text_sw

        elif last_clean_step == 1:
            clean_text = df_tweet_text_sw.lower()

        elif last_clean_step == 2:
            lower = df_tweet_text_sw.lower()
            clean_text = ' '.join([self.text_abbrevs.get(elem, elem) for elem in lower.split()])
        
        elif last_clean_step == 3:
            lower = df_tweet_text_sw.lower()
            without_text_abbrevs = ' '.join([self.text_abbrevs.get(elem, elem) for elem in lower.split()])
            clean_text = ' '.join([self.grammar_abbrevs.get(elem, elem) for elem in without_text_abbrevs.split()])
        
        elif last_clean_step == 4:
            lower = df_tweet_text_sw.lower()
            without_text_abbrevs = ' '.join([self.text_abbrevs.get(elem, elem) for elem in lower.split()])
            without_grammar_abbrevs = ' '.join([self.grammar_abbrevs.get(elem, elem) for elem in without_text_abbrevs.split()])
            
            joined_re_groups = '|'.join([group for group in self.re_substitution_groups])
            clean_text = ' '.join([re.sub(joined_re_groups,' ',word) for word in without_grammar_abbrevs.split()])
        
        elif last_clean_step == 5:
            lower = df_tweet_text_sw.lower()
            without_text_abbrevs = ' '.join([self.text_abbrevs.get(elem, elem) for elem in lower.split()])
            without_grammar_abbrevs = ' '.join([self.grammar_abbrevs.get(elem, elem) for elem in without_text_abbrevs.split()])
            
            joined_re_groups = '|'.join([group for group in self.re_substitution_groups])
            without_re_groups = ' '.join([re.sub(joined_re_groups,' ',word) for word in without_grammar_abbrevs.split()])

            clean_text = re.sub(r'\W',' ',without_re_groups)

        elif last_clean_step == 6:
            lower = df_tweet_text_sw.lower()
            without_text_abbrevs = ' '.join([self.text_abbrevs.get(elem, elem) for elem in lower.split()])
            without_grammar_abbrevs = ' '.join([self.grammar_abbrevs.get(elem, elem) for elem in without_text_abbrevs.split()])
            
            joined_re_groups = '|'.join([group for group in self.re_substitution_groups])
            without_re_groups = ' '.join([re.sub(joined_re_groups,' ',word) for word in without_grammar_abbrevs.split()])

            without_nontext = re.sub(r'\W',' ',without_re_groups)

            clean_text = ' '.join([word for word in without_nontext.split() if word not in stopwords])
        
        # words_greater_than_two_char = ' '.join([word for word in clean_text.split() if len(word) >= 3])

        one_space_separated_tweet = ' '.join([word for word in clean_text.split()])

        return one_space_separated_tweet

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TweetTokenizer

df=pd.read_csv('C:/Users/Trevor/Downloads/archive/political_social_media.csv',engine='python')


def my_tokenizer(doc):
    """Tokenizes document using RegExpTokenizer
    Args:
        doc: string
    Returns:
        list: tokenized words
    """

    tokenizer = RegexpTokenizer(r'\w+')
    # tokenizer= TweetTokenizer()
    article_tokens = tokenizer.tokenize(doc.lower())
    return article_tokens



df['message']=df['message'].apply(lambda x: 1 if x=='attack' else 0)


import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


df2=pd.read_json('C:/Users/Trevor/Downloads/concatenated_abridged.jsonl.gz',compression='gzip',lines=True)


# df2=df2[df2['lang'] =='en']

full_text=[]

for i in range(df2.shape[0]):
    try:
        if df2.iloc[i]['full_text'][:2] == 'RT':
            full_text.append(df2.iloc[i]['retweeted_status']['full_text'].split('http')[0])
        else:
            full_text.append(df2.iloc[i]['full_text'].split('http')[0])
    except:
        full_text.append('')
        # pass

full_text=[' '.join(i.splitlines()) for i in full_text]
full_text=[' '.join(i.split()) for i in full_text]

df_txt=list(df['text'])

df_txt=[i.split('http')[0].strip() for i in df_txt]

df_tmp=pd.DataFrame(columns=['tweets'],data=df_txt + full_text)
cleaner = TextCleaner()
df_tmp['tweets'] = df_tmp['tweets'].apply(lambda x: cleaner.clean_tweets(x, 6))
vectorizer = TfidfVectorizer(
    tokenizer=my_tokenizer,
    stop_words='english',
    max_features=5000)

mat = vectorizer.fit_transform(df_tmp['tweets']).toarray()
X_full=pd.DataFrame(data=mat)
# X['sent']=df['text'].apply(lambda x: analyser.polarity_scores(x)['neg'])
y=df['message']


X=X_full[:5000]

def cross_val(X,y,over_=True):

    rec_scores=[]
    prec_scores=[]
    f1_scores=[]

    for urgh in range(1):
        kf = KFold(n_splits=5,shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if over_==True:
                over = SMOTE(sampling_strategy='not majority')
                X_train,y_train=over.fit_resample(X_train,y_train)
                X_train=pd.DataFrame(columns=X_test.columns,data=X_train)


            model.fit(X_train,y_train)

            preds=model.predict(X_test)

            prec_scores.append(precision_score(y_test,preds))
            rec_scores.append(recall_score(y_test,preds))
            f1_scores.append(f1_score(y_test,preds))
            print (prec_scores)
    prec_res=np.mean(prec_scores)
    rec_res=np.mean(rec_scores)
    f1_res=np.mean(f1_scores)
    print ('Precision = {:.2f}, Recall = {:.2f}, F1 score = {:.2f}'.format(prec_res,rec_res,f1_res))
    return (np.mean(prec_scores),np.mean(rec_scores),np.mean(f1_scores))

from sklearn.naive_bayes import MultinomialNB

# from lightgbm import LGBMClassifier

# model=MultinomialNB()

# cross_val(X,y,over_=True)

models=['Logistic Regression','Multinomial Naive Bayes','Random Forest','XGBoost']
models_=[LogisticRegression(),MultinomialNB(),RandomForestClassifier(),XGBClassifier()]

model_scores=[]

for i in models_:
    model=i
    model_scores.append(cross_val(X,y,over_=True))

model=MultinomialNB()
over = SMOTE(sampling_strategy='not majority')
X_,y_=over.fit_resample(X,y)
model.fit(X_,y_)

preds=model.predict(X_full[5000:])

bad_tweets=np.array(full_text)[np.where(preds==1)]


# [analyser.polarity_scores(bad_tweets[i])['compound'] for i in range(bad_tweets.shape[0])]

bad_tweets
import json 

annots = json.load(open("C:/Users/Trevor/Downloads/first_half_annotated.json"))
hashes=list(annots['#presidentialcandidate'].values())
values=list(annots['0'].values())
annots1={hashes[i]:values[i] for i in range(len(hashes))}

annots2 = json.load(open("C:/Users/Trevor/Downloads/second_half_hashtags_annotated.json"))

annots1_df=pd.DataFrame.from_dict(annots1, orient='index')
annots2_df=pd.DataFrame.from_dict(annots2, orient='index')
annots=pd.concat([annots1_df,annots2_df])
pro_trump=list(annots[annots[0]==1].index)
pro_biden=list(annots[annots[0]==-1].index)
neutral=list(annots[annots[0]==0].index)

def clean_hash(x):
    x=['#' + i.replace('#','') for i in x]
    return x

pro_trump=clean_hash(pro_trump)
pro_biden=clean_hash(pro_biden)
neutral=clean_hash(neutral)
# def clean_tags(x):
    
scores=[]
for i in full_text:
    trump_score=0
    neutral_score=0
    biden_score=0
    for j in pro_trump:
        if j in i:
            trump_score+=1
    for k in neutral:
        if k in i:
            neutral_score+=1
    for l in pro_biden:
        if l in i:
            biden_score+=1
    scores.append((trump_score,neutral_score,biden_score))



scores_df=pd.DataFrame(columns=['pro_trump','neutral','pro_biden'],data=np.vstack(scores))
scores_df['total_score']=np.sum(np.vstack(scores),axis=1)
scores_df['attack']=preds
df_vader=pd.DataFrame.from_dict(analyser.polarity_scores(full_text[0]),orient='index').T
for i in range(len(full_text[1:])):
    df_vader_tmp=pd.DataFrame.from_dict(analyser.polarity_scores(full_text[i]),orient='index').T
    df_vader=pd.concat([df_vader,df_vader_tmp])
df_vader.columns=['vader_neg','vader_neu','vader_pos','vader_compound']
all_scores=pd.DataFrame(columns=list(scores_df.columns)+list(df_vader.columns),data=np.hstack((scores_df.values,df_vader.values)))
all_scores['text']=full_text
all_scores=all_scores[['text']+list(all_scores.columns[:-1])]
all_scores.drop(columns='total_score').to_csv('text+scores.csv')
# %%
all_scores
# %%
df2
# %%
annots
# %%
all_scores
# %%



all_scores[all_scores['vader_compound']!=0]
# %%
all_scores[(all_scores['attack']==1) | (all_scores['vader_compound']<0)]['text'].to_csv('tweets4.csv',index=False,encoding='utf-8-sig')
# %%
all_scores.drop(columns='total_score').to_csv('text+scores.csv',index=False,encoding='utf-8-sig')
# %%
import seaborn as sns
import matplotlib.pyplot as plt
fig,ax=plt.subplots()


ax=sns.kdeplot(all_scores['vader_neu'],label='Neutral')
ax=sns.kdeplot(all_scores['vader_neg'],label='Negative')
ax=sns.kdeplot(all_scores['vader_pos'],label='Positive')
ax.set_xlabel('Score')
ax.set_title('Vader scores')
ax.legend()
plt.tight_layout()
plt.savefig('V_comp.svg',bbox_inches='tight')
# %%
fig,ax=plt.subplots()
ax.set_xlabel('Score')
ax.set_title('Vader compound scores')
ax=sns.kdeplot(all_scores['vader_compound'],label='Compound')
plt.tight_layout()
plt.savefig('V_other.svg',bbox_inches='tight')
# %%
model_scores
#%%

mod_mat=np.vstack(model_scores)


# %%
import numpy as np
import matplotlib.pyplot as plt

fig,ax=plt.subplots(figsize=(6,7))

# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = mod_mat[:,0]#[12, 30, 1, 8, 22]
bars2 = mod_mat[:,1]#[28, 6, 16, 5, 10]
bars3 = mod_mat[:,2]#[29, 3, 24, 25, 17]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
ax.bar(r1, bars1,  width=barWidth, edgecolor='white', label='Precision')
ax.bar(r2, bars2,  width=barWidth, edgecolor='white', label='Recall')
ax.bar(r3, bars3,  width=barWidth, edgecolor='white', label='F1 score')
ax.set_title('5-fold CV score on labelling tweet as "attack" ')
ax.legend()
# Add xticks on the middle of the group bars
ax.set_xlabel('Model')
ax.set_ylabel('Score')
plt.xticks([r + barWidth for r in range(len(bars1))], models,rotation=90)
# fig.set_size_inches(18.5, 10.5)
plt.tight_layout()
plt.savefig('mod_results.svg')
# %%
fig,ax=plt.subplots()
dist2=pd.Series(preds).value_counts()
ax.bar(dist2.index,dist2)
ax.set_title('Number of tweets in given dataset predicted as attack')
plt.xticks([0,1],['not attack','attack'])
plt.tight_layout()
plt.savefig('attack_given.svg')
# %%
fig,ax=plt.subplots()
dist1=df['message'].value_counts()
ax.bar(dist1.index,dist1)
ax.set_title('Number of tweets in training data labelled as attack')
plt.xticks([0,1],['not attack','attack'])
plt.tight_layout()
plt.savefig('attack_training.svg')
# %%
