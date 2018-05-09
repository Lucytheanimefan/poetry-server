import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import newspaper
import os
from multiprocessing import Process

class SatireNewsScraper:
    def __init__(self, data_path = 'article_data/'):
        self.data_path = data_path

    def scrape_news(self, news_url='http://www.theonion.com/', article_type = 'news/onion'):
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
        self.test_data = []
        self.test_target_index = []
        self.targets = ['onion', 'cnn', 'huffpost', 'nytimes', 'yahoo']

    def generate_data(self, train_or_test):
        base_directory = 'article_data/' + train_or_test + '/news/'
        article_count = 0
        for i, outlet in enumerate(self.targets):
            directory = base_directory + outlet
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".txt"): # and article_count < 40: 
                    with open(os.path.join(directory, filename)) as file:
                        if train_or_test == 'train':
                            self.data.append(file.read().replace('\n', ''))
                            self.target_index.append(i)
                        else:
                            self.test_data.append(file.read().replace('\n', ''))
                            self.test_target_index.append(i)
 

    def train(self, classifier_type='svm'):
        text_clf = None
        # get term frequencies
        if classifier_type == 'naive_bayes':
            text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), 
            ('tfidf', TfidfTransformer()), 
            ('clf', MultinomialNB())])
        elif classifier_type == 'svm':
            text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
        elif classifier_type == 'grid_search':
            text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), 
            ('tfidf', TfidfTransformer()), 
            ('clf', MultinomialNB())])
            parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
            gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
            gs_clf = gs_clf.fit(self.data, self.target_index)
            self.text_clf = gs_clf
            return
        self.text_clf = text_clf.fit(self.data, self.target_index)

    def predict(self, data_to_classify, correct_values=[]):
        predicted = self.text_clf.predict(data_to_classify)
        if (len(correct_values) == len(data_to_classify)):
            performance = np.mean(predicted == correct_values)
            print('Performance:')
            print(performance)
            print('Predicted vs actual:')
            for i, predict in enumerate(predicted):
                print(self.targets[predict] + ', ' + self.targets[correct_values[i]])

        return predicted

    def performance(self):
        if self.text_clf is None:
            return
        predicted = self.text_clf.predict(self.test_data)
        print('Performance:')
        print(np.mean(predicted == self.test_target_index))
        for i, predict in enumerate(predicted):
            print(self.targets[predict] + ', ' + self.targets[self.test_target_index[i]])



if __name__ == '__main__':
    #targets = [2, 3, 0, 0]
    classifier = SatireNewsClassifier()
    classifier.generate_data('train')
    classifier.generate_data('test')
    classifier.train('svm')
    print('Trained')
    classifier.performance()
    
    # predictions = classifier.predict(test, targets)
    # print('Prediction:')
    # print(predictions)
    # for prediction in predictions:
    #     print(classifier.targets[prediction])

    scraper = SatireNewsScraper()
    #scraper.scrape_news()
    # p0 = Process(target=scraper.scrape_news, args=('http://www.bbc.com', 'test/news/bbc',))
    # p1 = Process(target=scraper.scrape_news, args=('http://www.foxnews.com', 'test/news/fox',))
    # p2 = Process(target=scraper.scrape_news, args=('https://www.yahoo.com/news/', 'test/news/yahoo',))
    # p3 = Process(target=scraper.scrape_news, args=('http://huffingtonpost.com', 'test/news/huffpost',))
    # p4 = Process(target=scraper.scrape_news, args=('https://www.nytimes.com', 'test/news/nytimes',))
    # p5 = Process(target=scraper.scrape_news, args=('http://cnn.com', 'test/news/cnn',))
    # p6 = Process(target=scraper.scrape_news, args=('http://washingtonpost.com', 'test/news/wapost',))
    # p7 = Process(target=scraper.scrape_news)
    # p0.start()
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    # p7.start()
    #scraper.scrape_news('http://www.foxnews.com', 'news/fox')
    #scraper.scrape_news('https://www.yahoo.com/news/', 'news/yahoo')