from preprocess import *
from index import *
import os
import json
import time
import ast
import collections

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
    pre, docnumbers = readFiles(path)
    # print(pre)
    next = index(pre)
    docvec,vectorizer = createDocumentVectors(pre)

    # loop over the files in the queries folder and store in a json
    for file in os.listdir(path2):
        filename, filenametxt = os.path.splitext(file)
        with open(os.path.join(path2, file), 'r') as f:
            text = f.read()
            textdic = clean2(text)
            queryvec = calculateQueryVector(textdic, vectorizer)
            results = retrieveAndRank(queryvec, docvec, vectorizer)
            
        with open('results.txt', 'a') as outfile:
            for key, (id, score) in enumerate(results[:1000],1):
                outfile.write(f" {filename} Q0 {id} {key} {score} Test Run\n")
                # outfile.close()



# if __name__ == "__main__":
#     main()

def timer():
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end - start}")

timer()