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
    print_vocabulary(next)

    # loop over the files in the queries folder and store in a json
    for file in os.listdir(path2):
        filename, filenametxt = os.path.splitext(file)
        with open(os.path.join(path2, file), 'r') as f:
            text = f.read()
            textdic = clean2(text)
            queryvec = calculateQueryVector(textdic, vectorizer)
            results = retrieveAndRank(queryvec, docvec, vectorizer)
            fin.extend([(int(filename), id, rank, score) for rank, (id, score) in enumerate(results[:1000],1)])
            
            fin.sort()

        # with open('results.txt', 'a') as outfile:
        #     for key, (id, score) in enumerate(results[:1000],1):
        #         outfile.write(f" {filename} Q0 {id} {key} {score} TestRun\n")
        #         # outfile.close()
        with open('results.txt', 'a') as out:
            for filename, id, rank, score in fin:
                out.write(f"{filename} Q0 {id} {rank} {score} TestRun\n")

def print_vocabulary(inverted_index):
    with open('vocabulary.txt', 'w') as file:
        for x, (word, y) in enumerate(inverted_index.items()):
            file.write(f"{word}\n")
            if x == 99:  # Stop after writing the first 100 tokens
                break

# if __name__ == "__main__":
#     main()

def timer():
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end - start}")

timer()