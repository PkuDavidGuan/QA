import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
docnum=970862
vectorizer = TfidfVectorizer()
corpus = []
for docid in range(1, docnum+1):
    with open('cut_documents_new/Doc'+str(docid)+'_cut.pickle', 'rb') as dfile:
        doc_list = pickle.load(dfile)
        doc = ''
        for word in doc_list:
            doc += word + ' '
        corpus.append(doc)
        print('Document ' + str(docid)  + ' read')


doc_vectors = vectorizer.fit_transform(corpus)

print('Fit success.')
with open('vectorizer.dump', 'wb') as vfile:
    pickle.dump(vectorizer, vfile)

with open('doc_vectors.pickle', 'wb') as docvfile:
    pickle.dump(doc_vectors, docvfile)


#tfidf = vectorizer.fit_transform(corpus)

#word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
#weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

#for i in range(1):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
#    print("-------这里输出第",i,u"类文本的词语tf-idf权重------")
#    for j in range(len(word)):
#        print(word[j],weight[i][j])