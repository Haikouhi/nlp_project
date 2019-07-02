
""" Imports and logging : we start with our imports and get logging established: """
import gzip
import gensim 
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

""" Now, let's take a closer look at this data below by printing the first line. """
data_file="/home/haikouhi/simplon/nlp_project/wmc.csv.gz"
#data_file="/home/haikouhi/simplon/nlp_project/wwc.csv.gz"

with gzip.open ('/home/haikouhi/simplon/nlp_project/wmc.csv.gz', 'rb') as f:
#with gzip.open ('/home/haikouhi/simplon/nlp_project/wwc.csv.gz', 'rb') as f:

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
Where each list within the main list contains a set of tokens from a user tweet. 
Word2Vec uses all these tokens to internally create a vocabulary. After building the vocabulary, 
we just need to call train(...) to start training the Word2Vec model. Behind the scenes 
we are actually training a simple neural network with a single hidden layer. 
But, we are actually not going to use the neural network after training. 
Instead, the goal is to learn the weights of the hidden layer. 
These weights are essentially the word vectors that we’re trying to learn. """
model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)


""" output : the first example shows a simple case of looking up words similar to the word game. 
All we need to do here is to call the most_similar function and provide the word game as the 
positive example. This returns the top 3 similar words. """

w1 = "game"
print(model.wv.most_similar (positive=w1, topn=3))

# look up top 3 words similar to 'polite'
w1 = ["win"]
print(model.wv.most_similar (positive=w1,topn=3))

# look up top 3 words similar to 'france'
w1 = ["lose"]
print(model.wv.most_similar (positive=w1,topn=3))

# look up top 3 words similar to 'shocked'
w1 = ["france"]
print(model.wv.most_similar (positive=w1,topn=3))

# look up top 3 words similar to 'shocked'
w1 = ["best"]
print(model.wv.most_similar (positive=w1,topn=3))

# look up top 3 words similar to 'shocked'
w1 = ["worst"]
print(model.wv.most_similar (positive=w1,topn=3))

w1 = ["angry"]
print(model.wv.most_similar (positive=w1,topn=3))

# look up top 3 words similar to 'shocked'
w1 = ["fun"]
print(model.wv.most_similar (positive=w1,topn=3))

