"""# Package **Imports**"""

import nltk
nltk.download('wordnet')
#!pip install nltk  # in case corrector needs to install package

import gzip
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, tree
from numpy import random
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import os
import time
from scipy.stats import variation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import spmatrix as spm

"""# Data Import Functions"""

# script to parse comments from original text files. detects reviews and ratings, and saves into separate variables

def parse(filename):
  f = gzip.open(filename, "rt", encoding="utf-8")
  revs=[]
  rates=[]
  counter=0

  for l in f:
    #counter = counter + 1
    rev_flag = l.find('review/text')
    if rev_flag == 0:
      rev = l[13:-1]
      revs.append(rev)
    rate_flag = l.find('review/score')
    if rate_flag == 0:
      rate = l[14:-1]
      rates.append(rate)
  f.close()
  return revs, rates

# Reads original datasets, splits them into separate files based on rating

def saveData(filename):
    print('Parsing file...')
    cat_revs, cat_rates = parse(filename)  # run function on auto data

    print('Writing files...')

    myfile1 = open('/content/drive/My Drive/COMP551/Final Project/input/health_ratings_1.txt', 'w')
    myfile2 = open('/content/drive/My Drive/COMP551/Final Project/input/health_ratings_2.txt', 'w')
    myfile3 = open('/content/drive/My Drive/COMP551/Final Project/input/health_ratings_3.txt', 'w')
    myfile4 = open('/content/drive/My Drive/COMP551/Final Project/input/health_ratings_4.txt', 'w')
    myfile5 = open('/content/drive/My Drive/COMP551/Final Project/input/health_ratings_5.txt', 'w')
    for i, rate in enumerate(cat_rates):
        if rate == '1.0':
            myfile1.write('%s\n' % cat_revs[i])
        elif rate == '2.0':
            myfile2.write('%s\n' % cat_revs[i])
        elif rate == '3.0':
            myfile3.write('%s\n' % cat_revs[i])
        elif rate == '4.0':
            myfile4.write('%s\n' % cat_revs[i])
        elif rate == '5.0':
            myfile5.write('%s\n' % cat_revs[i])

    myfile1.close()
    myfile2.close()
    myfile3.close()
    myfile4.close()
    myfile5.close()

    return

# Reads rating-split dataset, removes reviews less than 100 characters, shuffles, writes to new files with specified number of reviews

def filterData(filename, count):

    file = open(filename, 'r')
    lines = file.readlines()

    lines_filtered = []
    for line in lines:
        if len(line) > 100:
            lines_filtered.append(line)

    lines = random.sample(lines_filtered, count)

    filename, file_ext = os.path.splitext(filename)
    filename_new = filename + '_filtered' + file_ext

    with open(filename_new, 'w') as f:
        for line in lines:
            f.write("%s" % line)

    return

# Reads filtered datasets, combines ones of the same category, adds ratings to end of reviews

def reconstructData(filename_new, filenames, ratings):
    with open(filename_new, 'w') as f:
        for filename, rating in zip(filenames, ratings):
            with open(filename, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    f.write("%s\t%d\n" % (line.rstrip(), rating))

    return

# Read from specified file, recategorizes to 'negative', 'neutral', or 'positive', returns training x and y sets

def createDataset(train_file):
    with open(train_file, 'r') as file:
        train = file.readlines()

    # get targets
    y_train = list(map(lambda x: int(x.strip()[-1]), train))
    y_train = np.asarray(y_train)

    # remove targets
    x_train = list(map(lambda x: x.strip().rstrip('12345').strip(), train))
    x_train = np.asarray(x_train)

    idx_1 = np.asarray(np.where(y_train == 1)).ravel()
    idx_2 = np.asarray(np.where(y_train == 2)).ravel()
    idx_3 = np.asarray(np.where(y_train == 3)).ravel()
    idx_4 = np.asarray(np.where(y_train == 4)).ravel()
    idx_5 = np.asarray(np.where(y_train == 5)).ravel()

    idx_neg = np.concatenate((idx_1, idx_2), axis=None)
    idx_neu = idx_3
    idx_pos = np.concatenate((idx_4, idx_5), axis=None)

    x_train_neg = x_train[idx_neg]
    y_train_neg = np.full(len(idx_neg), "negative")

    x_train_neu = x_train[idx_neu]
    y_train_neu = np.full(len(idx_neu), "neutral")

    x_train_pos = x_train[idx_pos]
    y_train_pos = np.full(len(idx_pos), "positive")

    x_train = np.concatenate((np.reshape(x_train_neg, (len(x_train_neg), 1)),
                              np.reshape(x_train_neu, (len(x_train_neu), 1)),
                              np.reshape(x_train_pos, (len(x_train_pos), 1))),
                             axis=0)

    y_train = np.concatenate((np.reshape(y_train_neg, (len(y_train_neg), 1)),
                              np.reshape(y_train_neu, (len(y_train_neu), 1)),
                              np.reshape(y_train_pos, (len(y_train_pos), 1))),
                             axis=0)

    return x_train, y_train

"""# Text Preprocessing Function"""

# use it like this..... x_train = text_preprocess(x_train)

def text_preprocess(data):
    data = np.array(data)
    data = data#[0]  # just take portion for debugging
    for i in range(len(data)): 
        temp_str = data[i][0]
        
        temp_str = temp_str.lower()                 # Converting to lowercase
        cleanr = re.compile('<.*?>')
        temp_str = re.sub(cleanr, ' ', temp_str)        #Removing HTML tags
        temp_str = re.sub(r'[?|!||"|#|,|.|:|/]',r' ',temp_str)
        temp_str = re.sub(r'[\'|\-|*)|(||/]',r'',temp_str)        #Removing Punctuations excepy ',' and '.'
        
        data[i][0] = temp_str 
    
    data = list(data)
    #Lemming
    lemmer=WordNetLemmatizer()
    data_preprocessed = data
    auto_revs_lemmered = []
    for i in range(len(data_preprocessed)):
        temp_sentence = data_preprocessed[i][0].replace(",", " ") # Replace ',' by space
        temp_sentence = temp_sentence.replace(".", " ") #Replace '.' by space
        temp_sentence_lemmered=[' '.join([lemmer.lemmatize(temp_sentence_words, 'v') for temp_sentence_words in temp_sentence.split(' ')])]
        temp_sentence_lemmered=[' '.join([lemmer.lemmatize(temp_sentence_words, 'a') for temp_sentence_words in temp_sentence_lemmered[0].split(' ')])]
        auto_revs_lemmered.append((temp_sentence_lemmered[0]))
    
    return np.array(auto_revs_lemmered)

"""# Baseline Reproduction"""

# Baseline reproduction Decision Tree classification

filename_train = "drive/My Drive/COMP551-Project/automotive_filtered.txt"
x_train_movies, y_train_movies = createDataset(filename_train)

v_bin = CountVectorizer(analyzer="word", binary=True, token_pattern=u"(?u)\\b\\w\\w\\w+\\b")
x_data = v_bin.fit_transform(x_train_movies.ravel())
y_data = y_train_movies

dt = tree.DecisionTreeClassifier()

params = {'criterion': ['entropy'],
          'max_depth': [20, 40, 60, 80, 100],
          'min_samples_split': [80, 100, 120, 140, 160, 180, 200, 220]}
clf = GridSearchCV(dt, params, cv=10, scoring='f1_macro')
x_train = x_data
y_train = y_data
clf.fit(x_train, y_train)
print('File: ', filename_train)
print('Best Param: ', clf.best_params_)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(x_data, y_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    dt = dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)

    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='macro'))
    recall_scores.append(recall_score(y_test, y_pred, average='macro'))
    f1_scores.append(f1_score(y_test, y_pred, average='macro'))

print("Accuracy: %0.3f (+/- %0.3f)" % (np.mean(accuracy_scores), np.std(accuracy_scores) * 2))
print("Precision: %0.3f (+/- %0.3f)" % (np.mean(precision_scores), np.std(precision_scores) * 2))
print("Recall: %0.3f (+/- %0.3f)" % (np.mean(recall_scores), np.std(recall_scores) * 2))
print("F-measure: %0.3f (+/- %0.3f)" % (np.mean(f1_scores), np.std(f1_scores) * 2))

filename_train = "/content/drive/My Drive/COMP551/Final Project/input/movies_filtered.txt"
x_train_movies, y_train_movies = createDataset(filename_train)

v_bin = CountVectorizer(analyzer="word", binary=True, token_pattern=u"(?u)\\b\\w\\w\\w+\\b")
x_data = v_bin.fit_transform(x_train_movies.ravel())
y_data = y_train_movies

"""```
# This is formatted as code
```
"""

print (filename_train[1:10])

"""filename_train 

Type - str
<br>len - 71

# Data Extraction and Preprocessing
"""

#auto_revs_og, auto_rates_og = parse("drive/My Drive/COMP551-Project/Automotive.txt.gz")
auto_revs_og, auto_rates = createDataset("drive/My Drive/COMP551-Project/automotive_filtered.txt")
auto_revs = text_preprocess(auto_revs_og)

# import pandas as pd                                 #for data manipulation and analysis
# import nltk                                         #Natural language processing tool-kit
# from nltk.corpus import stopwords                   #Stopwords corpus
# from nltk.stem import PorterStemmer 
# import re
# from nltk.stem.snowball import SnowballStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.stem import WordNetLemmatizer

# def text_preprocess(data):
#     data = np.array(data)
#     data = data#[0]  # just take portion for debugging
#     for i in range(len(data)): 
#         data[i] = data[i].lower()                 # Converting to lowercase
#         cleanr = re.compile('<.*?>')
#         data[i] = re.sub(cleanr, ' ', data[i])        #Removing HTML tags
#         data[i] = re.sub(r'[?|!||"|#|,|.|:|/]',r' ',data[i])
#         data[i] = re.sub(r'[\'|\-|*)|(||/]',r'',data[i])        #Removing Punctuations excepy ',' and '.'
#     data = list(data)
#     #Lemming
#     lemmer=WordNetLemmatizer()
#     data_preprocessed = data
#     auto_revs_lemmered = []
#     for i in range(len(data_preprocessed)):
#         temp_sentence = data_preprocessed[i].replace(",", " ") # Replace ',' by space
#         temp_sentence = temp_sentence.replace(".", " ") #Replace '.' by space
#         temp_sentence_lemmered=[' '.join([lemmer.lemmatize(temp_sentence_words, 'v') for temp_sentence_words in temp_sentence.split(' ')])]
#         temp_sentence_lemmered=[' '.join([lemmer.lemmatize(temp_sentence_words, 'a') for temp_sentence_words in temp_sentence_lemmered[0].split(' ')])]
#         auto_revs_lemmered.append((temp_sentence_lemmered))
    
#     return auto_revs_lemmered

# data_preprocessed = text_preprocess(auto_revs)

#importing CSV

import csv

with open("/content/drive/My Drive/COMP551-Project/data_preprocessed.csv", 'r') as f:
  reader = csv.reader(f)
  data_preprocessed = list(reader)

"""# TF - IDF"""

# split data into train, validation and test sets

# auto_revs_og_tf = np.array([str(i) for i in auto_revs_og])

#auto_train, auto_test, auto_train_rates, auto_test_rates = train_test_split(auto_revs_og_tf, auto_rates, train_size=0.7, test_size=0.3, shuffle=True) # if want unprocessed data
auto_train, auto_test, auto_train_rates, auto_test_rates = train_test_split(auto_revs, auto_rates, train_size=0.7, test_size=0.3, shuffle=True)
auto_train, auto_valid, auto_train_rates, auto_valid_rates = train_test_split(auto_train, auto_train_rates, train_size=0.7, test_size=0.3)


# perform TF-IDF and transform data into TF-IDF features

tf_idf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1))             # substantiate
vectors_train_idf = tf_idf_vectorizer.fit_transform(auto_train) # run tf-idf on training set
#print(tf_idf_vectorizer.get_feature_names())                    # print words

vectors_valid_idf = tf_idf_vectorizer.transform(auto_valid)
vectors_test_idf = tf_idf_vectorizer.transform(auto_test)       # transform test set in tf-idf

"""# Linear SVM"""

### try tunning decision tree with data from baseline reproduction data

# filename_train = "drive/My Drive/COMP551-Project/automotive_filtered.txt"
# x_train_check, y_train_check = createDataset(filename_train)
# #x_train_check = text_preprocess(x_train_check)

# v_bin = CountVectorizer(analyzer="word", binary=True, ngram_range = (1,3), token_pattern=u"(?u)\\b\\w\\w\\w+\\b")
# x_data_fold = v_bin.fit_transform(x_train_check.ravel())
# y_data_fold = y_train_check

# x_train_check, x_test_check, y_train_check, y_test_check = train_test_split(x_data_fold, y_data_fold, train_size=0.7, test_size=0.3, shuffle=True)
# x_train_check, x_valid_check, y_train_check, y_valid_check = train_test_split(x_train_check, y_train_check, train_size=0.3, test_size=0.3, shuffle=True)

# vectors_train_idf = x_train_check
# auto_train_rates = y_train_check
# vectors_valid_idf = x_valid_check
# auto_valid_rates = y_valid_check
# vectors_test_idf  = x_test_check
# auto_test_rates = y_test_check

### ### ###

start_time=time.time()
Cs=np.linspace(0,1,101)
Cs=Cs[1:-1]
lsvm_vld_f1s=[]
cvs=[]


for c in Cs:
        #lsvm=OneVsRestClassifier(svm.LinearSVC(C=c), n_jobs=-2)
        lsvm=svm.LinearSVC(C=c)
        lsvm.fit(vectors_train_idf, auto_train_rates.ravel())
        pred_vld=lsvm.predict(vectors_valid_idf)
        lsvm_vld_f1s.append(f1_score(auto_valid_rates.ravel(), pred_vld, average='weighted'))
        #k_fold_lsvm=cross_val_score(lsvm, vectors_train_idf, auto_train_rates.ravel(), cv=3)
        #cvs.append(variation(k_fold_lsvm))
    
best_f1_lsvm=np.max(lsvm_vld_f1s)
best_c=Cs[lsvm_vld_f1s.index(best_f1_lsvm)]
#lsvm=OneVsRestClassifier(svm.LinearSVC(C=best_c), n_jobs=-2)
lsvm=svm.LinearSVC(C=best_c)
lsvm.fit(vectors_train_idf, auto_train_rates.ravel())

pred_lsvm=lsvm.predict(vectors_test_idf)
#k_fold_lsvm=cross_val_score(lsvm, x_data_fold, y_data_fold.ravel(), cv=10, scoring='f1_weighted')
k_fold_lsvm=cross_val_score(lsvm, x_data_fold, y_data_fold.ravel(), cv=10, scoring='f1_weighted')

duration=time.time() - start_time
perf_lsvm=classification_report(auto_test_rates.ravel(), pred_lsvm)
print("--- %s seconds ---" % (duration))

print(best_c)
print(perf_lsvm)
print('+/-',np.std(k_fold_lsvm))
#print(duration)

"""# Kernelized SVM"""

###  KERNELIZED SVM  ###

start_time=time.time()
Cs=np.linspace(0,1,51)   # C cost function, lecture 12 page 11
Cs=Cs[1:-1]
lsvm_vld_f1s=[]

for c in Cs:
        lsvm=OneVsRestClassifier(svm.SVC(C=c, kernel='rbf'), n_jobs=-2)
        lsvm.fit(vectors_train_idf, auto_train_rates.ravel())
        pred_vld=lsvm.predict(vectors_valid_idf)
        lsvm_vld_f1s.append(f1_score(auto_valid_rates.ravel(), pred_vld, average='weighted'))
    
best_f1_lsvm=np.max(lsvm_vld_f1s)
best_c=Cs[lsvm_vld_f1s.index(best_f1_lsvm)]
lsvm=OneVsRestClassifier(svm.SVC(C=best_c, kernel='rbf'), n_jobs=-2)
lsvm.fit(vectors_train_idf, auto_train_rates)

pred_lsvm=lsvm.predict(vectors_test_idf)
perf_lsvm=classification_report(auto_test_rates.ravel(), pred_lsvm)
print("--- %s seconds ---" % (time.time() - start_time))
print(best_c)
print(perf_lsvm)

"""# Gaussian Naive Bayes"""

start_time = time.time()
gnb=GaussianNB()
gnb.fit(spm.toarray(vectors_train_idf), auto_train_rates.ravel())
k_fold_gnb=cross_val_score(gnb, spm.toarray(vectors_train_idf), auto_train_rates.ravel(), cv=10, scoring='f1_weighted')

pred_gnb=gnb.predict(spm.toarray(vectors_test_idf))
perf_gnb = classification_report(auto_test_rates.ravel(), pred_gnb)
duration = time.time() - start_time

print(perf_gnb)
#print(best_)
print(np.mean(k_fold_gnb),'+/-',np.std(k_fold_gnb))
print(duration)

"""# Decision Tree"""

### try tunning decision tree with data from baseline reproduction
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.7, test_size=0.3, shuffle=True)

vectors_train_idf = x_train
auto_train_rates = y_train
vectors_valid_idf = x_valid
auto_valid_rates = y_valid
vectors_test_idf  = x_test
auto_test_rates = y_test
### ### ###

start_time=time.time()

max_depths = np.linspace(1, 32, 32, endpoint=True)
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)


dt_f1s=[]
all_dt_f1s=[]
all_all_dt_f1s=[]
all_all_all_dt_f1s=[]

for leaf in min_samples_leafs:
  for split in min_samples_splits:
    for depth in max_depths:
      dt = OneVsRestClassifier(tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth, min_samples_split=split, min_samples_leaf=leaf), n_jobs=-1)  # not c4.5 but CART (similar). choice of entropy based on https://stackoverflow.com/questions/34230063/can-we-choose-what-decision-tree-algorithm-to-use-in-sklearn
      dt = dt.fit(vectors_train_idf, auto_train_rates)
      pred_vld=dt.predict(vectors_valid_idf)
      dt_f1s.append(f1_score(auto_valid_rates, pred_vld, average='weighted'))
      
    all_dt_f1s.append(dt_f1s)
    dt_f1s=[]
    
  all_all_dt_f1s.append(all_dt_f1s)
  all_dt_f1s=[]

all_all_dt_f1s = np.array(all_all_dt_f1s)
  
best_f1_dt = np.max(all_all_dt_f1s)
#best_depth=max_depths[dt_f1s.index(best_f1_dt)]
best_idx = np.unravel_index(all_all_dt_f1s.argmax(), all_all_dt_f1s.shape)

best_depth = max_depths[best_idx[2]]
best_split = min_samples_splits[best_idx[1]]
best_leaf = min_samples_leafs[best_idx[0]]

dt = OneVsRestClassifier(tree.DecisionTreeClassifier(criterion="entropy", max_depth=best_depth, min_samples_split=best_split, min_samples_leaf=best_leaf), n_jobs=-1)
dt.fit(vectors_train_idf, auto_train_rates)
k_fold_dt=cross_val_score(dt, vectors_train_idf, auto_train_rates.ravel(), cv=10, scoring='f1_weighted')

pred_dt = dt.predict(vectors_test_idf)
perf_dt = classification_report(auto_test_rates, pred_dt)

duration=time.time() - start_time

print(best_depth, best_split, best_leaf)
print(perf_dt)
print(np.mean(k_fold_dt),'+/-',np.std(k_fold_dt))
print(duration)

n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 4)]
print(n_estimators)

"""# Random Forest"""

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 4)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True]

grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()

start_time = time.time()

rf_random = GridSearchCV(rf, grid, cv = 3, verbose=1, n_jobs = -1)
rf_random.fit(vectors_train_idf, auto_train_rates.ravel())
#k_fold_rf=cross_val_score(rf_random, vectors_train_idf, auto_train_rates.ravel(), cv=10, scoring='f1_weighted')
k_fold_lsvm=cross_val_score(rf_random, x_data_fold, y_data_fold.ravel(), cv=10, scoring='f1_weighted')

#rf.fit(vectors_train_idf, auto_train_rates.ravel())

#pred_rf = rf.predict(vectors_test_idf)
#perf_rf = classification_report(auto_test_rates, pred_rf)
#print(perf_rf)
duration = time.time() - start_time

pred_rf = rf_random.predict(vectors_test_idf)
perf_rf = classification_report(auto_test_rates, pred_rf)

print(rf_random.best_params_)
print(perf_rf)
print(np.mean(k_fold_rf),'+/-',np.std(k_fold_rf))
print(duration)

print(rf_random.best_params_)
print(perf_rf)
print(np.mean(k_fold_rf),'+/-',np.std(k_fold_rf))
print(duration)
