from preprocess import readFiles
from index import *
import os
import json

def sizeofcoll(path):
    return len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path,file))])

def getqueries(path):
    with open(path, 'r') as f:
        text = f.read()
    return text

def main():
    path = './coll/'
    path2 = './queries/'
    size = sizeofcoll(path)
    pre = readFiles(path)
    next = index(pre)
    docvec = createDocumentVectors(pre,size)
    # loop over the files in the queries folder and store in a json
    for file in os.listdir(path2):
        with open(os.path.join(path2, file), 'r') as f:
            text = f.read()
            queryvec = calculateQueryVector(text, next, size)
            results = retrieveAndRank(queryvec, next, docvec)

        with open('results.txt', 'w') as outfile:
            for k, v in enumerate(results[:1000], 1):
                outfile.write(k + ' ' + str(v) + '\n')
    
    outfile.close()
            



    
    print(next)


if __name__ == "__main__":
    main()

