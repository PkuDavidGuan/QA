import sys

infile = open(sys.argv[1], "rb")
for i in range(0, int(sys.argv[3])):
    line = infile.readline().strip('\n')
    if i >= int(sys.argv[2]):
        print line

infile.close()