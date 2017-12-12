import requests
import lxml.html
import urllib
import pickle
import time
import random
import sys
from bs4 import BeautifulSoup as BS

def google_search(question):
    word = question.encode(encoding='utf-8', errors='strict')
    baseUrl = 'https://www.google.com/search'

    data = {'q':word, 'oq':word, 'ie':'utf-8'}
    data = urllib.parse.urlencode(data)
    url = baseUrl+'?'+data

    try:
        html = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        print(e.code)
    except urllib.error.URLError as e:
        print(e.reason)

    google = BS(html, "lxml")
    div = google.find_all('div', attrs={'class': 'g'})
    print(len(div))
    corpus = []
    for d in div:
        raw_span = d.find_all('span', attrs={'class':'st'})
        print(len(raw_span))
        corpus.append(raw_span[0].get_text())

    return corpus

def bing_search(question):
    time.sleep(random.uniform(2,4))

    word = question.encode(encoding='utf-8', errors='strict')
    baseUrl = 'https://cn.bing.com/search'

    data = {'q':word}
    data = urllib.parse.urlencode(data)
    url = baseUrl+'?'+data

    try:
        html = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        print(e.code)
    except urllib.error.URLError as e:
        print(e.reason)

    bing = BS(html, "lxml")
    div = bing.find_all('div', attrs={'class': 'b_caption'})
    corpus = []
    for d in div:
        p = d.find_all('p')
        for i in p: 
            corpus.append(i.get_text())

    return corpus

def baidu_search(question):
    time.sleep(random.uniform(2,4))

    word = question.encode(encoding='utf-8', errors='strict')
    baseUrl = 'http://www.baidu.com/s'
    page = 1 #第几页

    data = {'wd':word,'pn':str(page-1)+'0','tn':'baidurt','ie':'utf-8','bsst':'1'}
    data = urllib.parse.urlencode(data)
    url = baseUrl+'?'+data
    
    try:
        html = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        print(e.code)
    except urllib.error.URLError as e:
        print(e.reason)
    baidu = BS(html, "lxml")
    
    td = baidu.find_all(class_="f")
    corpus = []
    for t in td:
        raw_data = []
        raw_font = t.find_all("font", attrs={'size': '-1'})
        for font in raw_font:
            words = font.get_text().split('\t')
            for word in words:
                raw_data.append(word)
        
        max_len = 0
        line = ''
        for i in range(len(raw_data)):
            data = raw_data[i].strip()
            if data != '' and max_len < len(data):
                max_len = len(data)
                line = data
                
        corpus.append(line)
    return corpus

def crawler(question):
    corpus1 = baidu_search(question)
    #corpus1 = google_search(question)
    corpus2 = bing_search(question)
    return corpus2+corpus1     

if __name__ == "__main__":
    with open('test.txt', 'r') as infile:
        count = 0
        while True:
            line = infile.readline().strip()
            if not line:
                break
            if count < int(sys.argv[1]):
                count += 1
                continue
            
            filename = 'online/question'+str(count)+'.pkl'
            with open(filename, 'wb') as outfile:
                print(str(count) + '\t' + line)
                pickle.dump(crawler(line), outfile)
                print('Done')
            
            count += 1
            if count%300 == 0:
                time.sleep(60)
# if __name__ == "__main__":
#     corpus = crawler('''人们常用哪趟公交车来形容步行''')
#     for c in corpus:
#         print(c)