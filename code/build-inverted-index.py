import pickle
docnum=970862
vocabulary2index_list = {}
for docid in range(1, docnum+1):
    with open('cut_documents_new/Doc'+str(docid)+'_cut.pickle', 'rb') as dfile:
        doc_list = pickle.load(dfile)
        for word in doc_list:
            if word not in vocabulary2index_list:
                vocabulary2index_list[word] = []
            vocabulary2index_list[word].append(docid)
with open('vocabulary2inverted_index.pickle', 'wb') as vfile:
    pickle.dump(vocabulary2index_list, vfile)
