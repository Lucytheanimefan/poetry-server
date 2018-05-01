import requests
from bs4 import BeautifulSoup
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])

class PoetrySources:
    OK_NUMBERS = ["89000","43000"]

    def __init__(self):
        self.source = "https://www.poetryfoundation.org/poems"
        self.urls = []
        self.poems = []
        self.concatenated_poems = []
        self.vectorizer = CountVectorizer()
        self.tfid_vectorizer = TfidfVectorizer()
        self.line_counts = []


    def find_urls(self, limit = 100):
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"}
        for i in range(limit):
            url = "%s/%i" % (self.source, 43000+i)
            #print(url)
            # Make the request
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.text, "html.parser")
            #print(soup.prettify())
            poem = soup.find("div", {"class":"o-poem"})
            if poem is not None:
                poem_text = poem.find_all("div")
                if poem_text is not None and len(poem_text)>0:
                    #print("Not none!")
                    self.urls.append(url)
                    poem_arr = [poem.text for poem in poem_text] # array of lines
                    for line in poem_arr:
                        self.concatenated_poems.extend([word.lower().strip() for word in re.findall(r'\S+|\n',line + ' \\n')])
                    # self.concatenated_poems.extend([line.split() for line in poem_arr])
                    self.poems.append(poem_arr)
                    self.line_counts.append(len(poem_text))

        #print(self.urls)
        #print(self.poems)

    def markov_chain(self, n_words=30):
        pairs = make_pairs(self.concatenated_poems)
        word_dict = {}
        for word_1, word_2 in pairs:
            if word_1 in word_dict.keys():
                word_dict[word_1].append(word_2)
            else:
                word_dict[word_1] = [word_2]
        first_word = np.random.choice(self.concatenated_poems)
        chain = [first_word]
        for i in range(n_words):
            chain.append(np.random.choice(word_dict[chain[-1]]))
        return ' '.join(chain)


    def tokenize_poems(self):
        for poem in self.poems:
            self.vectorizer.fit(poem)
            self.tfid_vectorizer.fit(poem)
            print(self.tfid_vectorizer.vocabulary_)
            #print(self.tfid_vectorizer.idf_)
            #vocab = self.vectorizer.get_feature_names() 
            #print(vocab)
            print("")
            #print(self.vectorizer.vocabulary_)


class PoetryParser:

    def __init__(self, url):
        r = requests.get(url)
        # print(r.text)
        self.soup = BeautifulSoup(r.text, "html.parser")
        self.url = url

    def parse(self):
        return self.soup.find_all("pre")[0]

    def parse_found(self):
        return self.soup.find_all("div", {"class": "o-poem"})


if __name__=="__main__":
    #p = PoetryParser("https://www.poets.org/poetsorg/poem/fly-0");
    # p1 = PoetryParser("https://www.poetryfoundation.org/poems/46304/retrospect-56d226248e844");
    #print(p.parse())
    sources = PoetrySources()
    sources.find_urls(10)
    print('MARKOV')
    poem = sources.markov_chain()
    print(poem)
    #sources.tokenize_poems()
    #sources.find_urls(10)

