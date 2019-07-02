# https://towardsdatascience.com/latent-semantic-analysis-sentiment-classification-with-python-5f657346f6a3

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
#%matplotlib inline


df = pd.read_csv('Reviews.csv')
df.head()

#print(df)


####
# TF-IDF is an information retrieval technique that weighs a term’s frequency (TF) 
# and its inverse document frequency (IDF). Each word has its respective TF and IDF score. *
# The product of the TF and IDF scores of a word is called the TFIDF weight of that word.
# Put simply, the higher the TFIDF score (weight), the rarer the word and vice versa.


tfidf = TfidfVectorizer()
tfidf.fit(df['Text'])
#print(tfidf)

X = tfidf.transform(df['Text'])
df['Text'][1]
#print(df)


# we can check out tf-idf scores for a few words within this sentence :
#print([X[1, tfidf.vocabulary_['peanuts']]])
#print([X[1, tfidf.vocabulary_['jumbo']]])
#print([X[1, tfidf.vocabulary_['error']]])
# Among the three words, “peanut”, “jumbo” and “error”, tf-idf gives the highest weight 
# to “jumbo”. Why? This indicates that “jumbo” is a much rarer word than “peanut” and “error”. 
# This is how to use the tf-idf to indicate the importance of words or terms inside a collection 
# of documents.



####
# Sentiment Classification : To classify sentiment, we remove neutral score 3, 
# then group score 4 and 5 to positive (1), and score 1 and 2 to negative (0). 
# After simple cleaning up, this is the data we are going to work with.
df.dropna(inplace=True)
df[df['Score'] != 3]
df['Positivity'] = np.where(df['Score'] > 3, 1, 0)
cols = ['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 'Summary']
df.drop(cols, axis=1, inplace=True)
#print(df.head())

# Train Test Split :

X = df.Text
y = df.Positivity
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_train),
#                                                                             (len(X_train[y_train == 0]) / (len(X_train)*1.))*100,
#                                                                             (len(X_train[y_train == 1]) / (len(X_train)*1.))*100))
# print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_test),
#                                                                             (len(X_test[y_test == 0]) / (len(X_test)*1.))*100,
#                                                                             (len(X_test[y_test == 1]) / (len(X_test)*1.))*100))


####
# You may have noticed that our classes are imbalanced, and the ratio of negative to positive instances is 22:78. 
# One of the tactics of combating imbalanced classes is using Decision Tree algorithms, so, 
# we are using Random Forest classifier to learn imbalanced data and set class_weight=balanced .
# First, define a function to print out the accuracy score.

def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #print("accuracy score: {0:.2f}%".format(accuracy*100))
    return accuracy

# To have efficient sentiment analysis or solving any NLP problem, we need a lot of features. 
# Its not easy to figure out the exact number of features are needed. 
# So we are going to try, 10,000 to 30,000. And print out accuracy scores associate with the number of features.
cv = CountVectorizer()
rf = RandomForestClassifier(class_weight="balanced")
n_features = np.arange(10000,30001,10000)

# def nfeature_accuracy_checker(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=rf):
#     result = []
#     print(classifier)
#     print("\n")
#     for n in n_features:
#         vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
#         checker_pipeline = Pipeline([
#             ('vectorizer', vectorizer),
#             ('classifier', classifier)
#         ])
#         print("Test result for {} features".format(n))
#         nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)
#         result.append((n,nfeature_accuracy))
#     return result

# tfidf = TfidfVectorizer()
# print("Result for trigram with stop words (Tfidf)\n")
# feature_result_tgt = nfeature_accuracy_checker(vectorizer=tfidf,ngram_range=(1, 3))


# Before we are done here, we should check the classification report : 

cv = CountVectorizer(max_features=30000,ngram_range=(1, 3))
pipeline = Pipeline([
        ('vectorizer', cv),
        ('classifier', rf)
    ])
sentiment_fit = pipeline.fit(X_train, y_train)
y_pred = sentiment_fit.predict(X_test)

#print(classification_report(y_test, y_pred, target_names=['negative','positive']))


# Chi-Squared for Feature Selection : Feature selection is an important problem in Machine learning. 
# I will show you how straightforward it is to conduct Chi square test based feature selection on our large scale data set.
# We will calculate the Chi square scores for all the features and visualize the top 20, 
# here terms or words or N-grams are features, and positive and negative are two classes. 
# given a feature X, we can use Chi square test to evaluate its importance to distinguish the class.
tfidf = TfidfVectorizer(max_features=30000,ngram_range=(1, 3))
X_tfidf = tfidf.fit_transform(df.Text)
y = df.Positivity
chi2score = chi2(X_tfidf, y)[0]

plt.figure(figsize=(12,8))
scores = list(zip(tfidf.get_feature_names(), chi2score))
chi2 = sorted(scores, key=lambda x:x[1])
topchi2 = list(zip(*chi2[-20:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.5)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')
plt.show();