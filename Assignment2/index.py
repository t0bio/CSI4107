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

def index_frequency(d): 
    tokens = dict()
    frequency = dict()
    counter = dict()
    for key in d: # each doc in dictionary
        counter[key] = Counter(d[key])
        words = d.get(key)

        for word in words:
            if word not in words:
                tokens.append(word)
                
    for token in tokens:
        frequency[token] = {}
    
    for key in counter:
        count = counter.get(key)
        for x in count:
            freq = count.get(x)
            frequency[x][key] = freq
    return frequency

def index_weighted(d):
    weighted = index_frequency(d)

    for key in weighted:
        frequency = weighted[key]

        maxFrequency = 0
        for key in frequency:
            if frequency[key] > maxFrequency:
                maxFrequency = frequency[key]

        idf = math.log((float(len(weighted))/len(frequency)), 2)

        for key in frequency:
            wordFrequency = float(frequency[key])/maxFrequency
            frequency[key] = idf * wordFrequency
    
    return weighted