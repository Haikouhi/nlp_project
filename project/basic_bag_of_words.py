# -*- coding: utf-8 -*-
"""basic_bag_of_words.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hdMmhqbxKLSVKp7-vPjGgx2SnpHGARsL
"""

import json

# j'ouvre mon JSON et je recupere que le texte des tweets que je stock dans une liste
tweets = []
with open('like.js', 'r') as f:
    favorites = json.load(f)
    for t in favorites :
        tweets.append(t["like"]["fullText"])
        
print(len(tweets))

# je transforme mon ensemble de tweet en une seule "phrase" que je met tout en minuscule
tweets_flatten = ''.join(tweets)
tweets_flatten = tweets_flatten.lower()
tweets_flatten = tweets_flatten.replace('#', '')
tweets_flatten = tweets_flatten.replace('@', '')
print(len(tweets_flatten))

# je créer une liste où chaque élément est un mot
tweets_flatten_splitted = tweets_flatten.split()

print(len(tweets_flatten_splitted))

# je definie les stop words et les supprime de ma data
english_stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
french_stop_words = ["au","aux","avec","ce","ces","dans","de","des","du","elle","en","et","eux","il","je","la","le","leur","lui","ma","mais","me","même","mes","moi","mon","ne","nos","notre","nous","on","ou","par","pas","pour","qu","que","qui","sa","se","ses","son","sur","ta","te","tes","toi","ton","tu","un","une","vos","votre","vous","c","d","j","l","à","m","n","s","t","y","été","étée","étées","étés","étant","suis","es","est","sommes","êtes","sont","serai","seras","sera","serons","serez","seront","serais","serait","serions","seriez","seraient","étais","était","étions","étiez","étaient","fus","fut","fûmes","fûtes","furent","sois","soit","soyons","soyez","soient","fusse","fusses","fût","fussions","fussiez","fussent","ayant","eu","eue","eues","eus","ai","as","avons","avez","ont","aurai","auras","aura","aurons","aurez","auront","aurais","aurait","aurions","auriez","auraient","avais","avait","avions","aviez","avaient","eut","eûmes","eûtes","eurent","aie","aies","ait","ayons","ayez","aient","eusse","eusses","eût","eussions","eussiez","eussent","ceci","celà","cet","cette","ici","ils","les","leurs","quel","quels","quelle","quelles","sans","soi"]
custom_stop_words = ["&amp;", "#", "@", "-", "–", "!", "?", ":)", "it's"]

tweets_flatten_splitted = [x for x in tweets_flatten_splitted if x not in english_stop_words and x not in french_stop_words and x not in custom_stop_words]

print(len(tweets_flatten_splitted))

# j'itere sur chaque mot (donc sur chaque element de ma list) et je calcul le nombre d'occurent EXACT de celui dans la phrase
wordfreq = []
for w in tweets_flatten_splitted:
    wordfreq.append(tweets_flatten_splitted.count(w))
print(len(wordfreq))

# je fais une association clef valeur pour le mot et l'occurence de celui-ci
result = dict((zip(tweets_flatten_splitted, wordfreq)))
print(result["kubernetes"])

# je "trie" bon dictionnaire par valeur afin d'avoir les mots ayant la plus grand occurence en premier
sorted_x = sorted(result.items(), key=lambda x: x[1], reverse=True)

print(sorted_x)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

#wordcloud = WordCloud(width=1600, height=800, max_words=2000).generate(" ".join(flat_text))
wordcloud = WordCloud().generate_from_frequencies(result)

plt.figure(figsize=(30, 30))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
