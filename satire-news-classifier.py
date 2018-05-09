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
        self.targets = ['onion', 'cnn', 'huffpost', 'nytimes', 'yahoo']

    def generate_data(self):
        base_directory = 'article_data/news/'
        article_count = 0
        for i, outlet in enumerate(self.targets):
            directory = base_directory + outlet
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".txt"): # and article_count < 40: 
                    with open(os.path.join(directory, filename)) as file:
                        self.data.append(file.read().replace('\n', ''))
                        self.target_index.append(i)
                        article_count += 1

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

    def performance(self, data_array_to_classify):
        if self.text_clf is None:
            return
        predicted = self.text_clf.predict(data_array_to_classify)
        np.mean(predicted == self.target_index)


if __name__ == '__main__':
    test = ['A group of 109 retired and former career and non-career U.S. ambassadors sent the Senate a letter Wednesday to express “serious concern” over the nomination of Gina Haspel to be CIA director due to her controversial involvement in the agency’s torture program.“We have no reason to question Ms. Haspel’s credentials as both a leader and an experienced intelligence professional. Yet she is also emblematic of choices made by certain American officials in the wake of the attacks of September 11, 2001 that dispensed with our ideals and international commitments to the ultimate detriment of our national security,” the ambassadors, who served in both Republican and Democratic administrations, wrote in advance of Haspel’s confirmation hearing Wednesday. “What we do know, based on credible, and as yet uncontested reporting, leaves us of the view that [Ms. Haspel] should be disqualified from holding cabinet rank.”', 
    'President Trump announced on Wednesday that North Korea had freed three American prisoners, removing a bitter and emotional obstacle ahead of a planned meeting between him and the young leader of the nuclear-armed nation.The release of the three prisoners, all American citizens of Korean descent, was a diplomatic victory for Mr. Trump and in some ways the most tangible gesture of sincerity shown by North Korea’s leader, Kim Jong-un, to improve relations with the United States after nearly seven decades of mutual antagonism.',
    'Iranian Scientist Annoyed He Has To Go Back To Shitty Old Job Building Nuclear Weapons ISFAHAN, IRAN—In the wake of President Trump’s announcement Tuesday that the United States would pull out of the international agreement to limit the Middle Eastern country’s program, Iranian nuclear scientist Ali Khatami was reportedly annoyed that he would have to return to his shitty old job building nuclear weapons. “Great, just what I wanted to do—go back to converting yellowcake into uranium hexafluoride all fucking day,” said a visibly irritated Khatami, adding that he wasn’t looking forward to being holed up in a newly reopened underground bunker, working overtime on the development of long-range ballistic missiles the way he always was before the Joint Comprehensive Plan of Action was ratified. “This totally sucks. I hate the work, and the hours are fucking awful. And I’ll bet they’re bringing back my old dickhead boss who’s never satisfied no matter how much weapons-grade U-235 we crank out. I had just found a nice new position at a small research lab closer to my family, too. Oh, well, it’s got to get done, so back to the goddamn grind.” In related news, American nuclear scientist David Ebeling reported feeling pretty irritated that his weapons-production facility’s output goals had been raised yet again.','SAN FRANCISCO—Touting the device’s state-of-the-art ability to incentivize exercise through intimidation, Fitbit released a new tracking collar Tuesday that tightens every second the person wearing it is inactive. “Whether you’re a fitness guru or a first-time runner, this sleek new wearable tech will jumpstart any routine by clamping around your throat and slowly restricting air from passing through your windpipe any time you take a break,” said the company’s CEO James Park, who added that for many customers, the fear of strangulation is just what they need to adopt a more active lifestyle. “To use the device, all you’ll have to do is secure the unbreakable lock around your neck, turn on the pulse monitor, and start moving. Don’t stop for any reason, though, because pausing for 10 seconds is more than enough time for the collar to leave you writhing on the ground, dying of hypoxia.” At press time, Park added that although it technically was exercise, the device would not log any movements affiliated with trying to rip off the collar.']

    targets = [2, 3, 0, 0]
    classifier = SatireNewsClassifier()
    classifier.generate_data()
    classifier.train('grid_search')
    print('Trained')
    predictions = classifier.predict(test, targets)
    print('Prediction:')
    print(predictions)
    # for prediction in predictions:
    #     print(classifier.targets[prediction])

    scraper = SatireNewsScraper()
    #scraper.scrape_news()
    # scraper.scrape_news('http://www.bbc.com', 'news/bbc')
    # p1 = Process(target=scraper.scrape_news, args=('http://www.foxnews.com', 'news/fox',))
    # p2 = Process(target=scraper.scrape_news, args=('https://www.yahoo.com/news/', 'news/yahoo',))
    # p1.start()
    # p2.start()

    # p3 = Process(target=scraper.scrape_news, args=('http://huffingtonpost.com', 'news/huffpost',))
    #p4 = Process(target=scraper.scrape_news, args=('https://www.nytimes.com', 'news/nytimes',))
    #p3.start()
    
    #p4.start()
    #scraper.scrape_news('http://www.foxnews.com', 'news/fox')
    #scraper.scrape_news('https://www.yahoo.com/news/', 'news/yahoo')