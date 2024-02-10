# Step 2. [10 points] Indexing:  Build an inverted index, with an entry for each word in the vocabulary. You can use any appropriate data structure (hash table, linked lists, Access database, etc.). An example of possible index is presented below. Note: if you use an existing IR system, use its indexing mechanism.

# •       Input: Tokens obtained from the preprocessing module

# •       Output: An inverted index for fast access
from collections import defaultdict
import json
import os
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# counts how many times word appears in doc (TF):
def countWordsInLine(token, processedTokens):
  count = 0
  for x in processedTokens:
    if x == token:
      count = count + 1
  return float(count)

def index(d):
    token = dict()
    for key in d: # each doc in dictionary
        visited = set() # stores tokens that have been already visitied in the doc
        for word in d[key]:
            if word not in visited: 
                if word not in token: # if word is not already in token dic
                    token[word] = [(key, countWordsInLine(word, d[key]))] # tuple with doc ID and TF
                else:
                    token[word].append((key, countWordsInLine(word, d[key])))
                visited.add(word)
    return token



