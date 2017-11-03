import requests
from bs4 import BeautifulSoup


class PoetryParser:

    def __init__(self, url):
        r = requests.get(url)
        # print(r.text)
        self.soup = BeautifulSoup(r.text, 'html.parser')
        self.url = url

    def parse(self):
        return self.soup.find_all('pre')[0]

    def parse_found(self):
        return self.soup.find_all('div', {"class": "o-poem"})


if __name__=='__main__':
    p = PoetryParser('https://www.poets.org/poetsorg/poem/fly-0');
    # p1 = PoetryParser('https://www.poetryfoundation.org/poems/46304/retrospect-56d226248e844');
    print(p.parse())

