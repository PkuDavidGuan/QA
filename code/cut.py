import pickle
import jieba
import jieba.posseg as pseg
import re


wikifile = 'wiki.txt'
infile = open(wikifile, "r")
titles = []
documents = []
cur_title = ''
cur_doc = ''
prevline = ''
docid = 0
jieba.enable_parallel(4)
while True:
    line = infile.readline()
    #print('.', end='')
    if len(line) == 0:
        break
    if line[0]=='【' and len(prevline)<1:
        if len(cur_doc)>0 and len(cur_title)>0:
            cut_doc = jieba.lcut(cur_doc)
            titles.append(cur_title)
            #documents.append(cut_doc)
            docid += 1
            #print('')
            with open('cut_documents_new/Doc'+str(docid)+'_cut.pickle', 'wb') as docfile:
                pickle.dump(cut_doc, docfile)
            print('Document '+ str(docid) + ': '+ cur_title+' done')
            cur_doc = ''
        cur_title = line.strip('\n').strip('【').strip('】')
    else:
        cur_doc += line
    prevline = line.strip('\n')
for title in titles:
    print(title)
infile.close()
with open('titles.pickle.1', 'wb') as tfile:
    pickle.dump(titles, tfile)