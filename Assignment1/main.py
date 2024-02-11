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
    fin = []

    # loop over the files in the queries folder and store in a json
    for file in os.listdir(path2):
        with open(os.path.join(path2, file), 'r') as f:
            text = f.read()
            textdic = clean2(text)
            queryvec = calculateQueryVector(textdic, vectorizer)
            results = retrieveAndRank(queryvec, docvec, vectorizer)

    # print the results into results.txt, The file should have the following format, for the top-1000 results for each query/topic (the queries should be ordered in ascending order):
    #topic_id Q0 docno rank score tag
    for rank, (id, score) in enumerate(results[:1000],1):
        with open('results.txt', 'a') as f:
            f.write(f"1 Q0 {docnumbers[id]} {rank} {score} tag\n")
    

# if __name__ == "__main__":
#     main()

def timer():
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end - start}")

timer()