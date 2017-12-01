import pickle
import chardet

vector = {}
with open("vector.NegUnk900wName.dec", "rb") as infile:
    line = infile.readline()
    count = 0
    while True:
        line = infile.readline()
        if not line:
            break

        try:
            line = line.decode("gbk").strip('\n').strip('\r').split(' ')
        except:
            continue
        
        if len(line) != 102:
            continue
        count += 1
        word = line[0]
        vec = []
        for i in range(1, 101):
            vec.append(float(line[i]))
        
        vector[word] = vec
        if count % 10000 == 0:
            print(count)
    print("Total: " + str(count))
with open("wordvec.pkl", "wb") as outfile:
    pickle.dump(vector, outfile)