import pandas as pd
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

#read train&test data
train=pd.read_csv('data/train.csv')
test=pd.read_csv('data/test.csv')
test['label']='t'

#preprocessing the data --> cleansing
def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=:]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)

def cleanup(text):
    text = textClean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

#data prep
test=test.fillna(' ')
train=train.fillna(' ')

for i in range(len(train)):
    train.loc[i, 'text'] = cleanup((train.loc[i,'text']))
    train.loc[i, 'title'] = cleanup((train.loc[i,'title']))
    train.loc[i, 'author'] = cleanup((train.loc[i,'author']))

for i in range(len(test)):
    test.loc[i, 'text'] = cleanup((test.loc[i,'text']))
    test.loc[i, 'title'] = cleanup((test.loc[i,'title']))
    test.loc[i, 'author'] = cleanup((test.loc[i,'author']))
    
test['total']=test['title']+' '+test['author']+test['text']
train['total']=train['title']+' '+train['author']+train['text']

#tfidf
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(train['total'].values)
tfidf = transformer.fit_transform(counts)

targets = train['label'].values
logreg = LogisticRegression()
logreg.fit(counts, targets)

example_counts = count_vectorizer.transform(test['total'].values)
predictions = logreg.predict(example_counts)

#create a new dataframe
predi=pd.DataFrame(test["id"])

#put the prediction result into "label" column in dataframe predi
predi["label"] = predictions

#export dataframa to csv file
predi.to_csv('submit.csv', index=False)