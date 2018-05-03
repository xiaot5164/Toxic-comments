# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:15:33 2018

@author: Hongtao Liu
"""

import datetime
start = datetime.datetime.now()

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('train.csv',nrows=30000).fillna(' ')
#test = pd.read_csv('test.csv').fillna(' ')

train_text = train['comment_text']
#test_text = test['comment_text']
#all_text = pd.concat([train_text, test_text])
all_text=train_text

word_vectorizer = CountVectorizer(
 #   sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=10000,binary=True)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
#test_word_features = word_vectorizer.transform(test_text)

#char_vectorizer = CountVectorizer(
 #   sublinear_tf=True,
#    strip_accents='unicode',
#    analyzer='char',
#    stop_words='english',
#    ngram_range=(2, 6),
#    max_features=50000)
#char_vectorizer.fit(all_text)
#train_char_features = char_vectorizer.transform(train_text)
#test_char_features = char_vectorizer.transform(test_text)

#train_features = hstack([train_char_features, train_word_features])
#train_features=hstack([train_char_features])
train_features=hstack([train_word_features])
#test_features = hstack([test_char_features, test_word_features])

scores = []
#submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=5, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

 #   classifier.fit(train_features, train_target)
 #   submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))
end = datetime.datetime.now()
print (end-start)
#submission.to_csv('submission.csv', index=False)