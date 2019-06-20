# https://github.com/kavgan/nlp-in-practice/blob/master/word2vec/Word2Vec.ipynb

# if you have two words that have very similar neighbors 
# (i.e. the usage context is about the same), then these words are probably quite similar in 
# meaning or are at least highly related. 


""" Imports and logging : we start with our imports and get logging established: """
import gzip
import gensim 
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

""" Now, let's take a closer look at this data below by printing the first line. """
data_file="reviews_data.txt.gz"

with gzip.open ('reviews_data.txt.gz', 'rb') as f:
    for i,line in enumerate (f):
        print(line)
        break


""" Read files into a list : Now that we've had a sneak peak of our dataset, 
we can read it into a list so that we can pass this on to the Word2Vec model. 
I'm doing a mild pre-processing of the reviews using gensim.utils.simple_preprocess (line). 
This does some basic pre-processing such as tokenization, lowercasing, etc and returns back a 
list of tokens (words). """
def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    
    logging.info("reading file {0}...this may take a while".format(input_file))
    
    with gzip.open (input_file, 'rb') as f:
        for i, line in enumerate (f): 

            if (i%10000==0):
                logging.info ("read {0} reviews".format (i))
            # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess (line)

# read the tokenized reviews into a list
# each review item becomes a serries of words
# so this becomes a list of lists
documents = list (read_input (data_file))
logging.info ("Done reading data file")



""" Training the Word2Vec model : just instantiate Word2Vec and pass the reviews that 
we read in the previous step (the documents). So, we are essentially passing on a list of lists. 
Where each list within the main list contains a set of tokens from a user review. 
Word2Vec uses all these tokens to internally create a vocabulary. After building the vocabulary, 
we just need to call train(...) to start training the Word2Vec model. Behind the scenes 
we are actually training a simple neural network with a single hidden layer. 
But, we are actually not going to use the neural network after training. 
Instead, the goal is to learn the weights of the hidden layer. 
These weights are essentially the word vectors that we’re trying to learn. """
model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)


""" output : the first example shows a simple case of looking up words similar to the word dirty. 
All we need to do here is to call the most_similar function and provide the word dirty as the 
positive example. This returns the top 10 similar words. """
w1 = "dirty"
model.wv.most_similar (positive=w1)

# look up top 6 words similar to 'polite'
w1 = ["polite"]
model.wv.most_similar (positive=w1,topn=6)

# look up top 6 words similar to 'france'
w1 = ["france"]
model.wv.most_similar (positive=w1,topn=6)

# look up top 6 words similar to 'shocked'
w1 = ["shocked"]
model.wv.most_similar (positive=w1,topn=6)

# get everything related to stuff on the bed
w1 = ["bed",'sheet','pillow']
w2 = ['couch']
model.wv.most_similar (positive=w1,negative=w2,topn=10)


""" Similarity between two words in the vocabulary : returns the similarity between two words that 
are present in the vocabulary. Under the hood, the above three snippets computes the cosine 
similarity between the two specified words using word vectors of each """
# similarity between two different words
print(model.wv.similarity(w1="dirty",w2="smelly"))

# similarity between two identical words
print(model.wv.similarity(w1="dirty",w2="dirty"))

# similarity between two unrelated words
print(model.wv.similarity(w1="dirty",w2="clean"))

""" Find the odd one out """
# Which one is the odd one out in this list?
print(model.wv.doesnt_match(["cat","dog","france"]))

# Which one is the odd one out in this list?
print(model.wv.doesnt_match(["bed","pillow","duvet","shower"]))
