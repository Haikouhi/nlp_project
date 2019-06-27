import requests
import base64
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

client_key = '748JXlKv7PjqQqXtP7yU8W1aY'
client_secret = 'KR2QrxDHc8cK5jmnHsRpAn8vtVikf8AELTo94v5oCWaM4jDNRF'

def init_api() :
    key_secret = '{}:{}'.format(client_key, client_secret).encode('ascii')
    b64_encoded_key = base64.b64encode(key_secret)
    b64_encoded_key = b64_encoded_key.decode('ascii')
    base_url = 'https://api.twitter.com/'
    auth_url = '{}oauth2/token'.format(base_url)
    auth_headers = {
        'Authorization': 'Basic {}'.format(b64_encoded_key),
        'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
    }
    auth_data = {
        'grant_type': 'client_credentials'
    }
    auth_resp = requests.post(auth_url, headers=auth_headers, data=auth_data)
    access_token = auth_resp.json()['access_token']
    return access_token


def main() :
    access_token = init_api()
    headers = {
        'Authorization': 'Bearer {}'.format(access_token)
    }
    # add hashtag after %23
    #data = requests.get(" https://api.twitter.com/1.1/search/tweets.json?q=%23foot&lang=fr&result_type=mixed&count=100", headers=headers)
    data = requests.get(" https://api.twitter.com/1.1/search/tweets.json?q=%23griezmann&lang=en&result_type=mixed&count=100", headers=headers)

    # creating a full json file with data 
    with open('data.json', 'w') as outfile:
        json.dump(data.json(), outfile, ensure_ascii=False, indent=2)

    # putting data in a dictionary
    with open('data.json', 'r') as f:
        datas = json.load(f)
        # print(type(datas))
        # print(datas.keys())
        # print(datas["statuses"])
        data_tweet = datas["statuses"]

    # creating json file filled with tweets 
    with open('data_tweet.json', 'w') as outfile:
        json.dump(data_tweet, outfile, ensure_ascii=False, indent=2)

    # stocking the tweets in a list
    tweets = []
    with open('data_tweet.json', 'r') as f:
        tweetfoot = json.load(f)
        for t in tweetfoot:
            tweets.append(t["text"])
    print("Total tweets : ", len(tweets))

    # removing capital letters, @ or #
    tweets_flatten = ''.join(tweets)
    tweets_flatten = tweets_flatten.lower()
    tweets_flatten = tweets_flatten.replace('#', '')
    tweets_flatten = tweets_flatten.replace('@', '')
    print("Total tweets in lowercase : ", len(tweets_flatten))

    # creating a list where each element is a separate word
    tweets_flatten_splitted = tweets_flatten.split()
    print("Total flattened tweets : ", len(tweets_flatten_splitted))

    # defining stops words
    english_stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                          "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                          "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
                          "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
                          "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
                          "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
                          "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
                          "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
                          "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
                          "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
                          "can", "will", "just", "don", "should", "now"]
    french_stop_words = ["au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", "eux", "il", "je",
                         "la", "le", "leur", "lui", "ma", "mais", "me", "même", "mes", "moi", "mon", "ne", "nos", "notre",
                         "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur",
                         "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "vous", "c", "d", "j", "l",
                         "à", "m", "n", "s", "t", "y", "été", "étée", "étées", "étés", "étant", "suis", "es", "est",
                         "sommes", "êtes", "sont", "serai", "seras", "sera", "serons", "serez", "seront", "serais",
                         "serait", "serions", "seriez", "seraient", "étais", "était", "étions", "étiez", "étaient", "fus",
                         "fut", "fûmes", "fûtes", "furent", "sois", "soit", "soyons", "soyez", "soient", "fusse", "fusses",
                         "fût", "fussions", "fussiez", "fussent", "ayant", "eu", "eue", "eues", "eus", "ai", "as", "avons",
                         "avez", "ont", "aurai", "auras", "aura", "aurons", "aurez", "auront", "aurais", "aurait",
                         "aurions", "auriez", "auraient", "avais", "avait", "avions", "aviez", "avaient", "eut", "eûmes",
                         "eûtes", "eurent", "aie", "aies", "ait", "ayons", "ayez", "aient", "eusse", "eusses", "eût",
                         "eussions", "eussiez", "eussent", "ceci", "celà", "cet", "cette", "ici", "ils", "les", "leurs",
                         "quel", "quels", "quelle", "quelles", "sans", "soi"]
    custom_stop_words = ["&amp", '&amp;', "#", "@", "-", '«', ':', '»']

    tweets_flatten_splitted = [x for x in tweets_flatten_splitted if
                               x not in english_stop_words and x not in french_stop_words and x not in custom_stop_words]

    print("Total w/o stop words : ", len(tweets_flatten_splitted))

    # calculating the number of occurrences of each word
    wordfreq = []
    for w in tweets_flatten_splitted:
        wordfreq.append(tweets_flatten_splitted.count(w))
    print("Total occurrences : ", len(wordfreq))

    # associating key-value pour words and their occurrences 
    result = dict((zip(tweets_flatten_splitted, wordfreq)))

    # filing through the dictionary by values in order of most occurrence first
    sorted_x = sorted(result.items(), key=lambda x: x[1], reverse=True)
    print(sorted_x)
    print(type(result))
    # wordcloud = WordCloud(width=1600, height=800, max_words=2000).generate(" ".join(flat_text))
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(result)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('foot.png', bbox_inches='tight')


if __name__ == '__main__':
    main()