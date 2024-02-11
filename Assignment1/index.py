# Step 2. [10 points] Indexing:  Build an inverted index, with an entry for each word in the vocabulary. You can use any appropriate data structure (hash table, linked lists, Access database, etc.). An example of possible index is presented below. Note: if you use an existing IR system, use its indexing mechanism.

# •       Input: Tokens obtained from the preprocessing module

# •       Output: An inverted index for fast access
from collections import defaultdict, Counter
import heapq
import json
import os
import re
import string
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import readFiles

# counts how many times word appears in doc (TF):
# def countWordsInLine(token, processedTokens):
#   count = 0
#   for x in processedTokens:
#     if x == token:
#       count = count + 1
#   return float(count)

# inv index -> works fine
def index(d): 
    token = dict()
    for key in d: # each doc in dictionary
        n = Counter(d[key])
        for word, count in n.items():
            if word not in token:
                token[word] = [(key, count)]
            else:
                token[word].append((key, count))
    # print(token)
    return token

def findMaxFrequency(processedTokens): # calculate max frequency of word in lsit of tokens
    return max(Counter(processedTokens).values())

def createDocumentVectors(collection):
    vectorizer = TfidfVectorizer()
    docvec = vectorizer.fit_transform([' '.join(collection[key]) for key in collection])
    return docvec, vectorizer

def calculateQueryVector(query, vectorizer):
    queryvec = vectorizer.transform([' '.join(query)])
    return queryvec

# def retrieveAndRank(queryvec, docvec, vectorizer):
#     similarity = cosine_similarity(queryvec, docvec)
#     results = list(enumerate(similarity[0]))
#     results.sort(key=lambda x: x[1], reverse=True)
#     return results

def retrieveAndRank(queryvec, docvec, vectorizer, docnumbers):
    results = []
    for num, vec in docvec.items():
        score = cosine_similarity(queryvec, vec)
        results.append((num, docnumbers[num], score))
    results.sort(key=lambda x: x[2], reverse=True)
    return results








