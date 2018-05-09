import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import newspaper
import os
from multiprocessing import Process

class SatireNewsScraper:
    def __init__(self, data_path = 'article_data/'):
        self.data_path = data_path

    def scrape_news(self, news_url='http://www.theonion.com/', article_type = 'satire'):
        print(news_url)
        news_object = newspaper.build(news_url, language='en')
        print('Built newspaper')
        articles = news_object.articles
        print(len(articles))
        previous_title = ''
        directory = self.data_path + article_type + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i, article in enumerate(articles):
            article.download()
            article.parse()
            if previous_title != article.title:
                with open(directory + str(i) + '.txt', 'w') as f:
                    previous_title = article.title
                    print(str(i) + ': ' + article.title)
                    f.write(article.title + '\n' + article.text)


# target names is a list
# data is a list (of lists?)
# target_index is a list of what the news is
class SatireNewsClassifier:
    def __init__(self, file_name = None):
        self.file_name = file_name
        self.data = []
        self.target_index = []
        self.targets = ['bbc', 'cnn', 'fox', 'wapost', 'yahoo']

    def generate_data(self):
        base_directory = 'article_data/news/'
        article_count = 0
        for i, outlet in enumerate(self.targets):
            directory = base_directory + outlet
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".txt") and article_count < 40: 
                    with open(os.path.join(directory, filename)) as file:
                        self.data.append(file.read().replace('\n', ''))
                        self.target_index.append(i)
                        article_count += 1
        print(self.target_index)
        print(type(self.data))
        print(type(self.targets))

    def train(self):
        # get term frequencies
        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), 
            ('tfidf', TfidfTransformer()), 
            ('clf', MultinomialNB())])
        self.text_clf = text_clf.fit(self.data, self.targets)

    def predict(data_to_classify):
        predicted = self.text_clf.predict(data_to_classify)
        return predicted

    def performance(data_array_to_classify):
        if self.text_clf is None:
            return
        predicted = self.text_clf.predict(data_array_to_classify)
        np.mean(predicted == self.target_index)


if __name__ == '__main__':
    huffposttest = ['A group of 109 retired and former career and non-career U.S. ambassadors sent the Senate a letter Wednesday to express “serious concern” over the nomination of Gina Haspel to be CIA director due to her controversial involvement in the agency’s torture program.“We have no reason to question Ms. Haspel’s credentials as both a leader and an experienced intelligence professional. Yet she is also emblematic of choices made by certain American officials in the wake of the attacks of September 11, 2001 that dispensed with our ideals and international commitments to the ultimate detriment of our national security,” the ambassadors, who served in both Republican and Democratic administrations, wrote in advance of Haspel’s confirmation hearing Wednesday. “What we do know, based on credible, and as yet uncontested reporting, leaves us of the view that [Ms. Haspel] should be disqualified from holding cabinet rank.”']
    classifier = SatireNewsClassifier()
    classifier.generate_data()
    classifier.train()
    print('Trained')
    prediction = classifier.predicted(huffposttest[0])
    print('Prediction:')
    print(prediction)
    print(classifier.targets[prediction])


    scraper = SatireNewsScraper()
    # scraper.scrape_news('http://www.bbc.com', 'news/bbc')
    # p1 = Process(target=scraper.scrape_news, args=('http://www.foxnews.com', 'news/fox',))
    # p2 = Process(target=scraper.scrape_news, args=('https://www.yahoo.com/news/', 'news/yahoo',))
    # p1.start()
    # p2.start()

    # p3 = Process(target=scraper.scrape_news, args=('http://huffingtonpost.com', 'news/huffpost',))
    # p4 = Process(target=scraper.scrape_news, args=('https://www.nytimes.com', 'news/nytimes',))
    # p3.start()
    # p4.start()
    #scraper.scrape_news('http://www.foxnews.com', 'news/fox')
    #scraper.scrape_news('https://www.yahoo.com/news/', 'news/yahoo')