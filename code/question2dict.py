import pickle

question = []
with open("raw_question.txt", 'r') as infile:
    while True:
        line = infile.readline().strip('\n').strip('\r')
        if not line:
            break
        pair = line.split('\t')
        question.append(pair)
        
with open("question.pkl", "wb") as outfile:
    pickle.dump(question, outfile)

