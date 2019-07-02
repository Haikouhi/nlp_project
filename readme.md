# How the twitterverse reacts to the Mens' World Championship vs the Womens' World Championship...


## Outils :

#### Word2vec :
- Méthode introduite par Thomas Mikolov, chercheur chez Google

- Research Paper “Efficient Estimation of Word Representations in Vector Space”, 2013

- Structure de réseau de neurones pour générer du word embedding en entraînant un model sur une problématique d’apprentissage supervisé pour une  classification.

- Réseau de neurone à deux couches
- Input : un corpus de texte, output : un set de vecteurs
- N’est pas du deep learning
- Objectif : Un set bien entraîné de vecteurs de mots va placer des mots similaires dans un même endroit de façon proche.


![](images/w2vec.png)

Elvis Costello “Writing about music is like dancing about architecture.”

*A est à B ce que C est à  ?*

*vector(king)	–	vector(man)	 +	vector(woman)	 =	 vector(queen)*


CBOW & Skip-gram:
![](images/w2vec2.png)

[Exemple de code ici :)](https://github.com/kavgan/nlp-in-practice/blob/master/word2vec/Word2Vec.ipynb)



*Understanding some of the parameters*

To train the model earlier, we had to set some parameters. Now, let's try to understand what some of them mean. For reference, this is the command that we used to train the model.

_model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=2, workers=10)_

- size: size of the dense vector to represent each token or word. If you have very limited data, then size should be a much smaller value. If you have lots of data, its good to experiment with various sizes. A value of 100-150 has worked well for me.

- window : maximum distance between the target word and its neighboring word. If your neighbor's position is greater than the maximum window width to the left and the right, then, some neighbors are not considered as being related to the target word. In theory, a smaller window should give you terms that are more related. If you have lots of data, then the window size should not matter too much, as long as its a decent sized window.

- min_count :minimium frequency count of words. The model would ignore words that do not statisfy the min_count. Extremely infrequent words are usually unimportant, so its best to get rid of those. Unless your dataset is really tiny, this does not really affect the model.

- workers : How many threads to use behind the scenes?



LSA :https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python

#### Topic Modeling :
A Topic Model can be defined as an unsupervised technique to discover topics across various text documents. These topics are abstract in nature, i.e., words which are related to each other form a topic. Similarly, there can be multiple topics in an individual document.




check the article out here :https://medium.com/@Haikouhi/comparing-tweets-between-the-mens-2018-and-women-s-2019-world-cups-with-natural-language-e74acebbfde9