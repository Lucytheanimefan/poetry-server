import requests
from bs4 import BeautifulSoup
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import sys
from random import *

def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])

class PoetrySources:
    OK_NUMBERS = [89000,43000]

    def __init__(self):
        self.source = "https://www.poetryfoundation.org/poems"
        self.urls = []
        self.poems = []
        self.concatenated_poems = []
        self.vectorizer = CountVectorizer()
        self.tfid_vectorizer = TfidfVectorizer()
        self.line_counts = []


    def populate_sources(self, limit = 100, start = int(random()*10)):
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"}
        for i in range(limit):
            url = "%s/%i" % (self.source, self.OK_NUMBERS[int(random())] + start + i)
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
                    self.poems.append(poem_arr)
                    self.line_counts.append(len(poem_text))

        print('--------Done populating sources---------')
        #print(self.urls)
        #print(self.poems)

    def markov_chain_poem(self, n_words=30):
        pairs = make_pairs(self.concatenated_poems)
        word_dict = {}
        for word_1, word_2 in pairs:
            if word_1 in word_dict.keys():
                word_dict[word_1].append(word_2)
            else:
                word_dict[word_1] = [word_2]
        first_word = np.random.choice(self.concatenated_poems)
        chain = [first_word]
        at_limit = False
        new_line = False
        i = 0
        while (not (at_limit and new_line)):
            if i == n_words:
                at_limit = True
            word = np.random.choice(word_dict[chain[-1]])
            if word == "\\n":
                new_line = True
            chain.append(word)
            i+=1
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


# class PoetryParser:

#     def __init__(self, url):
#         r = requests.get(url)
#         # print(r.text)
#         self.soup = BeautifulSoup(r.text, "html.parser")
#         self.url = url

#     def parse(self):
#         return self.soup.find_all("pre")[0]

#     def parse_found(self):
#         return self.soup.find_all("div", {"class": "o-poem"})


if __name__=="__main__":
    print(sys.argv)
    if len(sys.argv) >= 4:
        sources = PoetrySources()
        num_sources = int(sys.argv[1])
        start_index = int(sys.argv[2])
        length = int(sys.argv[3])
        sources.populate_sources(num_sources, start_index)
        print('MARKOV poem:')
        poem = sources.markov_chain_poem(100)
        print(poem)
    else:
        print("PoetryWriter.py <num_sources/poems> <start index> <length_of_poem>")
    #sources.tokenize_poems()
    #sources.find_urls(10)

