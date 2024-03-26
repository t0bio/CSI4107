import nltk
import string
import os
import json
import re

from sklearn import preprocessing

from bs4 import BeautifulSoup

from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer

def remove_tags(text):
    # Removes the HTML tags from the text
    soup = BeautifulSoup(text, "html.parser")
    strip= soup.get_text(separator=" ")
    return strip

def remove_punctuation(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ',text) # regex command to remove punctuation

def remove_numbers(text):
    return re.sub(r'\d', '', text) # regex command to remove numbers

def removeextrawhitespace(text):
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

def remove_stopwords(text):
    # Removes stopwords from text
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return filtered

def stem(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in text]

def dumpjson(dict, file):
    with open(file, 'w') as f:
        json.dump(dict, f)
        

# read files in for preprocessing and dump in json
def readFiles(path):
    v = dict()
    docnumbers = dict()
    for file in os.listdir(path):
        with codecs.open(os.path.join(path, file), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            docn = re.search(r'<DOCNO>(.*?)</DOCNO>', text)
            if docn:
                docn = docn.group(1)
                docnumbers[file] = docn
            text = text.lower()
            text = remove_numbers(text)
            text = remove_tags(text)
            text = remove_punctuation(text)
            text = removeextrawhitespace(text)
            text = remove_stopwords(text)
            text = stem(text)
            v[file] = text
            
    dumpjson(v, 'preprocessed.json')
    dumpjson(docnumbers, 'docnumbers.json')
    return v, docnumbers

    
    