from preprocess import readFiles
from index import *
import os
import json
import time


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
            for key, value in enumerate(results[:100], 1):
                outfile.write(f"1 Q0 {str(key)} {str(value[0])} {str(key)} {str(value[1])} testrun\n")

    outfile.close()


# if __name__ == "__main__":
#     main()

def timer():
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end - start}")

timer()