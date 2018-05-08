import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import newspaper
# target names is a list
# data is a list (of lists?)

class SatireNewsScraper:
	def __init__(self, data_path = 'article_data/'):
		self.data_path = data_path

	def scrape_satire(url='http://www.theonion.com/', article_type = 'satire'):
		onion = newspaper.build(url)
		print('Built onion')
		articles = onion.articles
		print(len(articles))
		for i, article in enumerate(articles):
			article.download()
			article.parse()
			with open(self.data_path + article_type + '/' + str(i) + '.txt', 'w') as f:
				print(article.title)
				f.write(article.title + '\n' + article.text)

class SatireNewsClassifier:
	def __init__(self, file_name = None):
		self.file_name = file_name

	def train(self, data, targets):
		# get term frequencies
		self.text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), 
			('tfidf', TfidfTransformer()), 
			('clf', MultinomialNB())])
		self.text_clf = text_clf.fit(data, targets)

	def predict(data_to_classify):
		predicted = self.text_clf.predict(data_to_classify)
		return predicted

	def performance(data_array_to_classify, target_index):
		if self.text_clf is None:
			return
		predicted = self.text_clf.predict(data_array_to_classify)
		np.mean(predicted == target_index)


if __name__ == '__main__':
	classifier = SatireNewsClassifier()
	scraper = SatireNewsScraper()
	scraper.scrape_satire()