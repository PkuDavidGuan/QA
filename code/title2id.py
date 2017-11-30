import pickle
with open('titles.pickle.1','rb') as tfile:
    titles = pickle.load(tfile)
    title2id = {}
    id = 0
    for title in titles:
        id += 1
        title2id[title] = id
    with open('title2id.pickle', 'wb') as tifile:
        pickle.dump(title2id, tifile)