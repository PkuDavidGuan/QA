# _*_coding=utf-8_*_
infile = open('Data/vectorsw300l20.all', 'r')
lines = infile.readlines()
word_to_vector_dict = {}
for line in lines[1:]:
    line = line.decode('utf-8')
    row = line.strip().split()
    word = row[0]
    vector = map(float, row[1:])
    word_to_vector_dict[word] = vector
