# Research Question

Can we generate

Try to determine what dataset something belongs to

# How our project benefits the community

Interesting, a case study in the dangers of Machine Learning, increase the temperature of the rhetoric.
 - 

# Modeling and Evaluation

Baseline: GPT-2 Model, 

Evaluation metric: Set of Generated examples as they differ across modelling datasets:
 - Intense/Trump
 - Non-intense/Trump
 - Intenst/Biden
 - Non-intense/Biden

Segmented based on manual annotation via hashtags. 

Subjectively, generate the tweets

Notes:
 - Considered unsupervised clustering of the text

Business area:
 - can it be applied to another area, for monetary gain



# Start

# Introduction

CRISP-DM process:

Research Q: 

Advances is Machine Learning text-generation: An exploration in inflammatory political social media posts

- Business Understanding

Slowly, everything in our everyday lives is going online. Roughly 2/5 [41%](https://www.wordstream.com/blog/ws/2019/04/04/ecommerce-statistics) of US consumers order 1-2 packages from Amazon per week and a whopping 83% have made one purchase from Amazon within the last 6 months. This transfer to digital-life has come to the fore during the COVID-19 pandemic. Work, social events, even education has transfered online with the focus of social distancing. Although beneficial to curb the pandemic, social isolationism makes it difficult to keep meaningful social relationships, and many result to communication through digital media.

Digital media is susceptible to misinformation, as have all become acutely aware in this election season, but couldn't humans spot the difference between artificially generated posts and those by humans? Honest and open dissemination of information is critical to American discourse, which pervaids all business transactions. The ability to artifically generate text for the purpose of increasing the temperature of civil discourse can have a significant impact on business public relations.


- Data Understanding/Data Preparation

Datasets:

1. ~140k tweets pertaining to 2020 Presidential Election [link](https://drive.google.com/file/d/1rBJBWWTF9lvKs4pY-PF9Wad91yiW45ol/view?usp=sharing)
2. 5k social media posts from politicians' social media accounts, along with human judgements to about the nature of the tweet, see list below [link](https://www.kaggle.com/crowdflower/political-social-media-posts)

  - attack: the message attacks another politician
  - constituency: the message discusses the politician's constituency
  - information: an informational message about news in government or the wider U.S.
  - media: a message about interaction with the media
  - mobilization: a message intended to mobilize supporters
  - other: a catch-all category for messages that don't fit into the other
  - personal: a personal message, usually expressing sympathy, support or condolences, or other personal opinions
  - policy: a message about political policy
  - support: a message of political support )

Our data consists of two datasets; one, a collection of raw micro-blog posts (tweets) with associated meta-data, and two, a collection of social media posts from politicians along with human judgements on the nature of the post (see above). The second dataset was used to train a machine learning model to detect 'attack/non-attack' judged social posts and applied to the first dataset. This provided a 'tweet intensity' feature to our initial dataset.  

Our plan was to curate 6 tweet subpopulation from dataset 1, spanning the political spectrum (pro-Trump/pro-Biden), the partisan intentsity spectrum (attack/non-attack), and sentiment spectrum (compound score, [VADER](https://github.com/cjhutto/vaderSentiment)). Political sentiment was judged by our team, manually annotating associated hashtags as either pro-Trump (+1), pro-Biden (-1) or neutral (0). After annotation, the hashtags associated with each tweet were summed to give a composite score, ranging from [-1, 1]. Partisan intensity of each tweet was decided via a custom attack/non-attack classifier that was trained on dataset 2 (detailed in models section), with a score of 0 - non-attack, and 1-attack. Sentiment was calculated with the VADER algorithm, a lexicon-based sentiment analyzer, providing a compound score ranging from [-1,1].

The segmented subpopulations of dataset 1 were used as training datasets for [GPT-2](https://github.com/openai/gpt-2), which generated new tweets that were aggregated to new datasets. 

Before segmenting our data for GPT-2 generation, we took a look at the data. For dataset 1, many of the tweets had neutral hashtags (~96%), with 2.5% favorible to biden, and 1.5% favoring trump. 

<p align="center">
<img src="imgs/support_of_candidates.png" width='400'/>
</p>

<p align="center">
<text> Histogram of human hashtag annotation for tweet political preference.</text>
</p>

Analysis of these tweets with VADER provided more insight to the distribution of sentimental laoding.

<p align="center">
<img src="imgs/V_comp.png" width='400'/>
</p>

<p align="center">
<text> Vader componenet scores. Positive, negative and neutral as a fraction of tokens in tweet.</text>
</p>

<p align="center">
<img src="imgs/V_other.png" width='400'/>
</p>

<p align="center">
<text> Vader compound sentiment score. The compound score takes token sentiment and context into consideration, ranging from negative, -1, to positive, 1.</text>
</p>

The distribution of attack/non-attack social media posts from our sourced dataset was roughly 1:5.

<p align="center">
<img src="imgs/attack_training.png" width='400'/>
</p>

<p align="center">
<text>Social media posts labeled attack/non-attack in our sourced dataset.</text>
</p>

Using our trained Multinomial Naive-Bayes model, we classified the tweets from dataset 1.

<p align="center">
<img src="imgs/attack_given.png" width='400'/>
</p>

<p align="center">
<text>Distribution of attack/non-attack labels given to dataset 1 from our classifier.</text>
</p>

- Modeling



Multinomial nieve Bayes classifier, used to classify as attack/non-attack



- Evaluation
- Deployment





# Code (50% of grade)

## General

- The project plan identifies criteria for successful completion of the project.
- README contains a concise project description and addresses each step in the CRISP-DM* process from business understanding to deployment 
- All assumptions are documented and next steps are laid out. 
- README and other project documentation is well-written, contains no spelling or grammatical errors, and ready for immediate review by potential employers 
- All files (including slides) are organized into directories and their locations are explained in the README 
- Final product addresses original business problem or research question.

## Use Case Understanding

- The project has the potential to provide benefit to users beyond just being interesting. (Is this project useful?) 
- Practical use is well articulated and demonstrated by project and project notes. (How well is this being explained as useful?) 
- The data set is put to good use in terms of practical application. (Is this the best use of your data set?)

## Data Prep

- The data preparation pipeline is documented in executable code contained in .py files. 

DATA SOURCE:
- Data source is clearly identified and credited. 
- Clear instructions are provided for obtaining the data and replicating the data preparation process. 
- Scripts are provided for obtaining the public data.
- A small representative shard of example data is included in the repository.

## Python Code

- All code is organized into functions and/or classes
- All Python code complies with PEP8 and passes pycode style. Docstrings are present and comply with Numpy or Google docstring style. Type annotations may be used. All inline comments are accurate and useful.
- All data preparation, modeling, and evaluation steps are documented in code. Another DSI graduate could reproduce the project’s results simply by following the instructions referenced in the README file.
- The repository is clean and does not contain unneccessary derived files and directories.  
- Modeling is done in .py files and not in jupyter notebooks

# Presentation (40%)

## Project Question/Goal

- Articulately and clearly describes the practical case of the project 
- Clearly articulates a question to be answered by the project  
- Identifies and articulates potential next steps

## EDA

- Identifies and describes the data source, crediting the data sources
- Demonstrates sufficient amount of analysis to properly identify data trends
- Demonstrates an understanding of the data as it relates to the project question

## Analysis (e.g. cleaning pipeline, modeling, validation of model, presentation of results)

- Chooses speaking points that are considered technologically interesting to the data science community; avoids topics that are considered common knowledge or uninteresting analysis
- Articulates metrics used for model validation 
- Presents modeling results in a way that is understandable to a technical and non-technical audience alike
- Final product addresses original business problem or research question.

## Visual Presentation & Presentation Skills

-------------------Visual Presentation-----------------
- Slides have no spelling or grammatical errors 
- Slide content is logically organized and easy to read/understand 
- Slides contain appropriate graphics to enhance understanding
- Slides enhance and do not detract from the project presentation (no meme rule) 
- Slides contain a link to student's github in summary 
-------------------Presentation Skills-----------------
- Maintains appropriate eye contact 
- Does not use a script 
- Does not have long pauses or excessive filler words 
- Demonstrates excitement and engagement with the topic 
- Demonstrates appropriate body language
- Always uses technical and professional communication and avoids slang/casual phrases 
- Technical content is factual and accurate
- Presentation does not exceed allotted 5 minute time limit

# Creativity (10%)

## Creativity Skills

Chooses points that are considered technologically interesting to the data science community; avoids topics that are considered common knowledge or uninteresting analysis

*Judge for Creativity has the liberty to award points at discretion*

Some Examples of Creativity:
- Original Strategies to the Data Science Project
- Unique Approaches to the Use of the Data 
- Original use of algorithms
- Novel use of libraries
- Unique and/or interactive visualizations