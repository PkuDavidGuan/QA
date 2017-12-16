import re
import crawler
import pickle

def question_clean(q):
    topic = ''
    poem = ''
    flag = 0
    if re.search('《', q):
        start = q.find('《')
        end = q.find('》')
        topic = q[start+1:end]
    
    if re.search('“', q):
        start = q.find('“')
        end = q.find('”')
        poem = q[start+1:end]
    
    next = re.compile('.*下一句.*|.*后半句.*|.*下句.*|.*后面一句.*|.*下半句.*|.*后一句.*|.*下半句.*')
    prev = re.compile('.*上一句.*|.*上句.*|.*前半句.*|.*前一句.*')

    if prev.match(q):
        flag = 1
    elif next.match(q):
        flag = 2
    return [topic, poem, flag]

def poem_answer(keyword, data):
    if keyword[0]=='' and keyword[1]=='':
        return None
    if keyword[2] == 0:
        return None

    query = keyword[0] + ' ' + keyword[1]
    corpus = data[keyword[1]]
    if not corpus:
        return None


    punctuation = r'“|”|，| |：|。|:|\?|\.|\"|？|_|-|[0-9]|[a-zA-Z]|,|\(|\)|\[|\]|{|}|\||\*|@|~|/|、|;|；|!|！'

    candidate = {}

    for line in corpus:
        start = line.find(keyword[1])
        if start == -1:
            continue
        if keyword[2] == 1:
            tmp = line[:start].strip()
            tmp = re.split(punctuation, tmp)
            for i in reversed(tmp):
                if i.strip() != '':
                    tmp = i.strip()
                    break
        else:
            tmp = line[start+len(keyword[1]):].strip()
            tmp = re.split(punctuation,tmp)
            for i in tmp:
                if i.strip() != '':
                    tmp = i.strip()
                    break
        if str(tmp) not in candidate.keys():           
            candidate[str(tmp)] = 1
        else:
            candidate[str(tmp)] += 1
    ans = sorted(candidate.items(), key=lambda d:d[1], reverse = True)
    
    p2 = re.compile('.*下一句.*|.*上一句.*|.*后半句.*|.*下句.*|.*上句.*|.*前半句.*|.*后面一句.*|.*下半句.*|.*后一句.*|.*下半句.*|.*前一句.*')
    for a in ans:
        if not p2.match(a[0]):
            return a[0]
    return None

def create_corpus(keyword):
    if keyword[0]=='' and keyword[1]=='':
        return None
    if keyword[2] == 0:
        return None

    query = keyword[0] + ' ' + keyword[1]
    if keyword[0] == '':
        query = '俗语 ' + query
    return crawler.crawler(query)

if __name__ == '__main__':
    with open('poem.pkl', 'rb') as infile:
        data = pickle.load(infile)
    with open('test.txt', 'r') as infile, open('open_final_out.txt', 'r') as file2:
        p2 = re.compile('.*下一句.*|.*上一句.*|.*后半句.*|.*下句.*|.*上句.*|.*前半句.*|.*后面一句.*|.*下半句.*|.*后一句.*|.*下半句.*|.*前一句.*')
        data_out = []
        flag = False
        while True:
            line = infile.readline().strip()
            line2 = file2.readline().strip()
            if not line:
                break

            if p2.match(line):
                print('---------------------------------------------')
                print(line)
                keyword = question_clean(line)
                answer = poem_answer(keyword, data)
                if answer:
                    flag = True
                    data_out.append(answer)
            if not flag:
                data_out.append(line2)
            flag = False
        with open('open_last_out.txt', 'w') as outfile:
            for d in data_out:
                outfile.write(d+'\n')

    


# if __name__ == '__main__':
#     data = {}
#     with open('test.txt', 'r') as infile:
#         p2 = re.compile('.*下一句.*|.*上一句.*|.*后半句.*|.*下句.*|.*上句.*|.*前半句.*|.*后面一句.*|.*下半句.*|.*后一句.*|.*下半句.*|.*前一句.*')
#         while True:
#             line = infile.readline().strip()
#             if not line:
#                 break
#             if p2.match(line):
#                 print('---------------------------------------------')
#                 print(line)
#                 keyword = question_clean(line)
#                 corpus = create_corpus(keyword)
#                 data[keyword[1]] = corpus
    
#     with open('poem.pkl', 'wb') as outfile:
#         pickle.dump(data, outfile)