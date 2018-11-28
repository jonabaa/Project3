# This code classifies the data differently
# https://www.kaggle.com/gloriahristova/a-walkthrough-eda-vizualizations-unigram-model/notebook

# Data processing
import pandas as pd
import json
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# Data vizualizations
import matplotlib.pyplot as plt

# Data Modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import own classified data
from RandomForest import *

train_data = pd.read_json('train.json') # store as dataframe objects

# Prepare the data
features = [] # list of list containg the recipes
for item in train_data['ingredients']:
    features.append(item)

ingredients = [] # this list stores all the ingredients in all recipes (with duplicates)
for item in train_data['ingredients']:
    for ingr in item:
        ingredients.append(ingr)

target = train_data['cuisine']

# Both train and test samples are processed in the exact same way
# Train
features_processed= [] # here we will store the preprocessed training features
for item in features:
    newitem = []
    for ingr in item:
        ingr.lower() # Case Normalization - convert all to lower case
        ingr = re.sub("[^a-zA-Z]"," ",ingr) # Remove punctuation, digits or special characters
        ingr = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', ingr) # Remove different units
        newitem.append(ingr)
    features_processed.append(newitem)

#Binary representation of the training set will be employed
vectorizer = CountVectorizer(analyzer = "word",
                             ngram_range = (1,1), # unigrams
                             binary = True, #  (the default is counts)
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_df = 0.99) # any word appearing in more than 99% of the sample will be discarded

# Fit the vectorizer on the training data and transform the test sample
X = vectorizer.fit_transform([str(i) for i in features_processed])

# Apply label encoding on the target variable (before model development)
lb = LabelEncoder()
Y = lb.fit_transform(target)

#split data
x_train1, x_test1, y_train1, y_test1 = train_test_split(X, Y, test_size = 0.2)

x_train2, x_test2, y_train2, y_test2 = easydatagen()

# Makes 31 random forests with 1 to 32 trees
# Plots the accuracies

"""
print('Starting training:')

testscores1 = []
testscores2 = []
index = []

for i in range(1,32):
    clf1 = RandomForestClassifier(n_estimators=i, max_depth=None, max_features='auto',
                             verbose=True, n_jobs=8)
    clf1.fit(x_train1, y_train1)
    clf2 = RandomForestClassifier(n_estimators=i, max_depth=None, max_features='auto',
                             verbose=True, n_jobs=8)
    clf2.fit(x_train2, y_train2)

    index.append(i)
    testscores1.append(clf1.score(x_test1, y_test1))
    testscores2.append(clf2.score(x_test2, y_test2))

plt.plot(index, testscores1, '-b' , label=' "PCA"-test data')
plt.plot(index, testscores2, '-r', label='Own test data')
plt.legend(loc='center right')
plt.title('Accuracy of random forests')
plt.xlabel('Trees')
plt.ylabel('Accuracy')
plt.show()
"""

"""
# 50 trær
print('Starting training:')

testscores1 = []
testscores2 = []
index = []

for i in range(1,51):
    clf1 = RandomForestClassifier(n_estimators=i, max_depth=None, max_features='auto',
                             verbose=True, n_jobs=8)
    clf1.fit(x_train1, y_train1)
    clf2 = RandomForestClassifier(n_estimators=i, max_depth=None, max_features='auto',
                             verbose=True, n_jobs=8)
    clf2.fit(x_train2, y_train2)

    index.append(i)
    testscores1.append(clf1.score(x_test1, y_test1))
    testscores2.append(clf2.score(x_test2, y_test2))

plt.plot(index, testscores1, '-b' , label=' "PCA"-test data')
plt.plot(index, testscores2, '-r', label='Own test data')
plt.legend(loc='center right')
plt.title('Accuracy of random forests')
plt.xlabel('Trees')
plt.ylabel('Accuracy')
plt.show()
"""
