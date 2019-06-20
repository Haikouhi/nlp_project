# https://www.analyticsvidhya.com/blog/2018/10/stepwise-guide-topic-modeling-latent-semantic-analysis/

# Let’s load the required libraries before proceeding with anything else.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_colwidth", 200)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import umap



from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
#documents = dataset.target_names

#print(len(documents))
#print(documents)


####
# Data Preprocessing :To start with, we will try to clean our text data as much as possible. 
# The idea is to remove the punctuations, numbers, and special characters all in one step using 
# the regex replace(“[^a-zA-Z#]”, ” “), which will replace everything, except alphabets with space. 
# Then we will remove shorter words because they usually don’t contain useful information. 
# Finally, we will make all the text lowercase to nullify case sensitivity.

news_df = pd.DataFrame({'document':documents})

# removing everything except alphabets`
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")

# removing short words
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# make all text lowercase
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())



####
# It’s good practice to remove the stop-words from the text data as they are mostly 
# clutter and hardly carry any information. Stop-words are terms like ‘it’, ‘they’, ‘am’, ‘been’, 
# ‘about’, ‘because’, ‘while’, etc.

#To remove stop-words from the documents, we will have to tokenize the text, i.e., 
# split the string of text into individual tokens or words. We will stitch the tokens back 
# together once we have removed the stop-words.
# how to install nltk.download here : https://www.nltk.org/data.html
stop_words = stopwords.words('english')

# tokenization
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())

# remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# de-tokenization
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df['clean_doc'] = detokenized_doc
#print(news_df)



#### Document-Term Matrix : This is the first step towards topic modeling. 
# We will use sklearn’s TfidfVectorizer to create a document-term matrix with 1,000 terms.
vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000, # keep top 1000 terms 
max_df = 0.5, 
smooth_idf=True)

X = vectorizer.fit_transform(news_df['clean_doc'])

X.shape # check shape of the document-term matrix

#print(vectorizer)
#print(X)



#### 
# Topic Modeling : The next step is to represent each and every term and document as a vector. 
# We will use the document-term matrix and decompose it into multiple matrices. 
# We will use sklearn’s TruncatedSVD to perform the task of matrix decomposition. 
# Since the data comes from 20 different newsgroups, let’s try to have 20 topics for our text data.
# The number of topics can be specified by using the n_components parameter.


# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)

svd_model.fit(X)

#print(len(svd_model.components_))



####
# The components of svd_model are our topics, and we can access them using svd_model.components_.
# Finally, let’s print a few most important words in each of the 20 topics and see how our model has done.
terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    #print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
        print(" ")


####
# Topics Visualization : To find out how distinct our topics are, we should visualize them. 
# Of course, we cannot visualize more than 3 dimensions, but there are techniques like PCA and t-SNE which can help us 
# visualize high dimensional data into lower dimensions. Here we will use a relatively new technique called UMAP 
# (Uniform Manifold Approximation and Projection).

X_topics = svd_model.fit_transform(X)
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)

plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1], 
c = dataset.target,
s = 10, # size
edgecolor='none'
)
plt.show()