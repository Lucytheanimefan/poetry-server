import requests
from bs4 import BeautifulSoup


class PoetryParser:

    def __init__(self, url):
        self.poetry_foundation = 'https://www.poets.org/'
        self.url = url

    def parse(self):
        r = requests.get(self.url)
        soup = BeautifulSoup(r.text, 'html.parser')
        return soup.find_all('pre')[0]




if __name__ == '__main__':
    p = PoetryParser('https://www.poets.org/poetsorg/poem/fly-0')
    print(p.parse())

