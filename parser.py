import requests
from bs4 import BeautifulSoup


class PoetrySources:
    OK_NUMBERS = ['89000','43000']

    def __init__(self):
        self.source = 'https://www.poetryfoundation.org/poems'
        self.urls = []

    def find_urls(self, limit = 100):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        for i in range(100):
            url = "%s/%i" % (self.source, 43000+i)
            print(url)
            # Make the request
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.text, 'html.parser')
            print(soup.prettify())
            poem = soup.find('div',{'class':'o-poem'}, text=True)
            if poem is not None:
                self.urls.append(url)
                print(poem)

        print(self.urls)


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
    #p = PoetryParser('https://www.poets.org/poetsorg/poem/fly-0');
    # p1 = PoetryParser('https://www.poetryfoundation.org/poems/46304/retrospect-56d226248e844');
    #print(p.parse())
    sources = PoetrySources()
    sources.find_urls(10)

