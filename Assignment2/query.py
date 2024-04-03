import os 
import sys
import json
import re
import nltk
from nltk.tokenize import word_tokenize
import string
import chardet
import time
from nltk.stem import PorterStemmer

# clean and stem the query, no need for beautifusoup
def cleanquery(text):
    stem = PorterStemmer()
    stop_words = set(line.strip('\n') for line in open("./StopWords.txt", "r"))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    cleaned_words = [stem.stem(word) for word in words if word not in stop_words and not word.isnumeric()]
    return cleaned_words

def jsonmake(dictionary, filename):
    with open(filename, "w") as outfile:
        json_str = json.dumps(dictionary)
        outfile.write(json_str)
    
def process_queries(directory):
    queries = dict()
    for filename in os.listdir(directory):
        if filename == ".DS_Store":
            continue
        rawdata = open(os.path.join(directory, filename), "rb").read()
        result = chardet.detect(rawdata)
        charenc = result['encoding']  
        with open(os.path.join(directory, filename), 'r', encoding=charenc) as file:
            query = file.read().strip()
            tokens = cleanquery(query)
            queries[filename] = tokens
    jsonmake(queries, "queries.json")
    return queries
    
def timer():
    start = time.time()
    process_queries("./queries")
    end = time.time()
    print(f"Time taken: {end-start} seconds")
timer()
            







