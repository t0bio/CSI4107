# Step 2. [10 points] Indexing:  Build an inverted index, with an entry for each word in the vocabulary. You can use any appropriate data structure (hash table, linked lists, Access database, etc.). An example of possible index is presented below. Note: if you use an existing IR system, use its indexing mechanism.

# •       Input: Tokens obtained from the preprocessing module

# •       Output: An inverted index for fast access
from collections import defaultdict, Counter
import heapq
import json
import os
import re
import string
import nltk
import math
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from multiprocessing import Pool

# counts how many times word appears in doc (TF):
# def countWordsInLine(token, processedTokens):
#   count = 0
#   for x in processedTokens:
#     if x == token:
#       count = count + 1
#   return float(count)

def index(d):
    token = dict()
    for key in d: # each doc in dictionary
        # visited = set() # stores tokens that have been already visitied in the doc
        # for word in d[key]:
        #     if word not in visited: 
        #         if word not in token: # if word is not already in token dic
        #             token[word] = [(key, countWordsInLine(word, d[key]))] # tuple with doc ID and TF
        #         else:
        #             token[word].append((key, countWordsInLine(word, d[key])))
        #         visited.add(word)
        n = Counter(d[key])
        for word, count in n.items():
            if word not in token:
                token[word] = [(key, count)]
            else:
                token[word].append((key, count))
    return token

def findMaxFrequency(processedTokens): # calculate max frequency of word in lsit of tokens
    # count = 0
    # for x in processedTokens:
    #     tmp = processedTokens.count(x)
    #     if tmp > count:
    #         count = tmp
    return max(Counter(processedTokens).values())
    
def createDocumentVectors(collection, size): # doc vectors
    def vec(line):
        doc, *tokens = line
        maxFrequency = findMaxFrequency(tokens)
        n = Counter(tokens)
        return doc, [(n, (count/maxFrequency) * math.log2(size/len(collection))) for n, count in n.items()]
    
    with Pool() as p:
        return dict(p.map(vec, collection))
    # weightedDict = dict()
    # for line in collection:
    #     weightedDict[line[0]] = []
    #     maxFrequency = findMaxFrequency(line[1:])
    #     # visited = []
    #     # for token in line[1:]:
    #     #     if token not in visited:
    #     #         tf_idf = (countWordsInLine(token, line[1:])/maxFrequency) * math.log2(size/(len(indexDict[token])))
    #     #         weightedDict[line[0]].append((token, tf_idf))
    #     #     visited.append(token)
    #     n = Counter(line[1:]) # number of occurences of words
    #     weightedDict[line[0]] = [(n, (count/maxFrequency) * math.log2(size/len(collection))) for n, count in n.items()]
    # return weightedDict

def calculateQueryVector(query, index, size):
    # queryVector = defaultdict(float)
    # queryTerms = set(query)
    # for term in queryTerms:
    #     if term in index:
    #         df = len(index[term])  # document frequency
    #         idf = math.log2(size / df) if df != 0 else 0
    #         tf_idf = (1 + math.log2(query.count(term))) * idf
    #         queryVector[term] = tf_idf
    # return queryVector
    queryVector = []
    queryTerms = Counter()
    for term in query:
        queryTerms[term] += 1
    for term, count in queryTerms.items():
        if term in index:
            df = len(index[term])
            idf = math.log2(size/df) if df != 0 else 0
            tf_idf = (1 + math.log2(count)) * idf
            queryVector.append(tf_idf)
    return queryVector

def cosine_similarity(v1, v2):# cosine similarity between two vecs
        sumx = 0
        sumy = 0
        sumxy = 0
        ans = 0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            sumx += x * x
            sumy += y * y
            sumxy += x * y
            ans = sumxy/math.sqrt(sumx * sumy)
            return ans 

def retrieveAndRank(query, invertedIndex, documentVectors):
    queryVector = calculateQueryVector(query, invertedIndex, len(documentVectors))
    results = []
    for docId, docVector in documentVectors.items():
        similarity = cosine_similarity(queryVector, docVector)
        
        if len(results) < 100:
            heapq.heappush(results, (docId, similarity))
        else:
            heapq.heappushpop(results, (docId, similarity))

    # Rank the results based on similarity scores in descending order
    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results








