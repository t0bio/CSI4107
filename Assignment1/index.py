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

def index(dict):
    token = dict()
    for key in dict:
        for word in dict[key]:
            if word not in token:
                token[word] = {key}
            else:
                token[word].add(key)
    