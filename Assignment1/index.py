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
import sys # delete after
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from multiprocessing import Pool

# counts how many times word appears in doc (TF):
def countWordsInLine(token, processedTokens):
  count = 0
  for x in processedTokens:
    if x == token:
      count = count + 1
  return float(count)

def index(d):
    indexDict = dict()
    for line, v in d.items():
        visited=[]
        for elem in v:
            if elem not in visited:
                if elem in indexDict:
                    indexDict[elem].append((line, countWordsInLine(elem, v)))
                else:
                    indexDict[elem]=[(line, countWordsInLine(elem, v))]
                visited.append(elem) 
    return indexDict

def findMaxFrequency(processedTokens): # calculate max frequency of word in lsit of tokens
    count = 0
    for x in processedTokens:
        tmp = processedTokens.count(x)
        if tmp > count:
            count = tmp

    return count
    # return max(Counter(processedTokens).values())
    
def createDocumentVectors(collection, index, size): # doc vectors
    # def vec(line):
    #     doc, *tokens = line
    #     maxFrequency = findMaxFrequency(tokens)
    #     n = Counter(tokens)
    #     return doc, [(n, (count/maxFrequency) * math.log2(size/len(collection))) for n, count in n.items()]
    
    # with Pool() as p:
    #     return dict(p.map(vec, collection))
    weightedDict = dict()
    for line, v in collection.items():
        weightedDict[line] = []
        maxFrequency = findMaxFrequency(v)
        visited = []
        for elem in v:      
            if elem not in visited:
                tf_idf = (countWordsInLine(elem, v)/maxFrequency) * math.log2(size/(len(index[elem])))
                weightedDict[line].append((elem, tf_idf))
            visited.append(elem)
            # n = Counter(line[1:]) # number of occurences of words
        # weightedDict[elem] = [(n, (count/maxFrequency) * math.log2(size/len(collection))) for n, count in n.items()]
    return weightedDict

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

def retrieval_ranking(test_query, inverted_index):
  # Setting up query vector
  topics = test_query.lower().split()
  queryVector = {}
  maxFrequency = findMaxFrequency(topics)
  
  #calculating query vector values
  for x in inverted_index:
    if x in topics:
      queryVector[x] = (0.5 + (0.5 * (countWordsInLine(x, topics)/maxFrequency))) * math.log2(size/(len(inverted_index[x])))
    else:
      queryVector[x]=0
  
  innerProduct = {}

  #inner product calculation
  for document_no in weightedDict:
    for x in weightedDict[document_no]:
      if document_no in innerProduct:
        innerProduct[document_no] = innerProduct[document_no] + (x[1] * queryVector[x[0]])
      else:   
        innerProduct[document_no] = x[1] * queryVector[x[0]]

  # Bottom part of COSINE SIM
  cosineSIM = {}
  queryWeights = 0
  docWeights = {}
  
  #w_iq summation
  for x in queryVector:
    queryWeights = queryWeights + (queryVector.get(x)**2)
  
  # w_ij summation:
  for document_no in weightedDict:
    total = 0
    weightsInDoc = weightedDict[document_no]
    for x in weightsInDoc:
      total = total + (x[1]**2)
    docWeights[document_no] = total
    
#cosine simularity
  for document_no in weightedDict:
    if document_no in innerProduct:
      cosineSIM[document_no] = innerProduct[document_no] / (math.sqrt(docWeights[document_no]) * math.sqrt(queryWeights))
    else:
      cosineSIM[document_no] = 0

  sort_cosineSIM = sorted(cosineSIM.items(), key=lambda x: x[1], reverse=True)
  return list(sort_cosineSIM)

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
    print(results)
    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results








