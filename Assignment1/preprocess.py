import nltk
import ssl
import pandas as pd
import pickle as pk
import numpy as np
import re
import unicodedata
import os
import warnings
import string
import json


# # nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')

from bs4 import BeautifulSoup

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')

# A part of this code was derived and developed from the following webpage tutorial: https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/#text-cleaning

# Reading in the documents for preprocessing from the coll folder
# cwd = os.getcwd()
# path = cwd + "/coll"
# files = os.listdir(path)
# os.chdir(path)

# paste this at the start of code

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

with open("./StopWords.txt", "r") as f:
    stop_words = f.read().splitlines()

# Preprocessing helper functions
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


def readFiles(path):
    v = dict()
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r') as f:
            text = f.read()
            text = text.lower()
            text = remove_numbers(text)
            text = remove_tags(text)
            text = remove_punctuation(text)
            text = removeextrawhitespace(text)
            text = remove_stopwords(text)
            text = stem(text)
            v[file] = text

    
    # write preprocessed files to a pickle file
    with open('./preprocessed.pickle', 'wb') as outfile:
        pk.dump(v, outfile, protocol=pk.HIGHEST_PROTOCOL)

    return v


# def readFiles(path, num): 
#     v = dict()
#     text = path
#     text = text.lower()
#     text = remove_tags(text)
#     text = remove_punctuation(text)
#     text = remove_numbers(text)
#     text = removeextrawhitespace(text)
#     text = remove_stopwords(text)
#     text = stem(text)
#     v[num] = text
#     return v

           
# def remove_numbers(text):
#     retru
