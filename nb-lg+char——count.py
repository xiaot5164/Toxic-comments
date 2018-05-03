# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:25:54 2018

@author: Hongtao Liu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 10:04:19 2018

@author: Hongtao Liu
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 13:45:44 2018

@author: Hongtao Liu
"""
import warnings
warnings.filterwarnings("ignore")
import datetime
start = datetime.datetime.now()

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
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
    max_features=10000,binary=False)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
#test_word_features = word_vectorizer.transform(test_text)

#char_vectorizer = CountVectorizer(

#    strip_accents='unicode',
#    analyzer='char',
 #   stop_words='english',
#    ngram_range=(2, 6),
#    max_features=50000)
#char_vectorizer.fit(all_text)
#train_char_features = char_vectorizer.transform(train_text)
#test_char_features = char_vectorizer.transform(test_text)




#train_features = hstack([train_word_features, train_char_features])
#train_features=hstack([train_char_features])
train_features=hstack([train_word_features])
#test_features = hstack([test_char_features, test_word_features])


#submission = pd.DataFrame.from_dict({'id': test['id']})
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from scipy import sparse
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p =  x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self




scores = []
#submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = NbSvmClassifier(C=4, dual=True, n_jobs=-1)
    cv_score = np.mean(cross_val_score(classifier,train_features, train_target, cv=5, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

 #   classifier.fit(train_features, train_target)
 #   submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))
end = datetime.datetime.now()
print (end-start)
#submission.to_csv('submission.csv', index=False)