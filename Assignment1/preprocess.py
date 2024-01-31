import nltk
import pandas as pd
import numpy as np
import re
import unicodedata
import os
import warnings

nltk.download('punkt')
# nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

from bs4 import BeautifulSoup

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
warnings.filterwarnings("ignore", category=DeprecationWarning)

# A part of this code was derived and developed from the following webpage tutorial: https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/#text-cleaning

# Reading in the documents for preprocessing from the coll folder
cwd = os.getcwd()
path = cwd + "/coll"
files = os.listdir(path)
os.chdir(path)
with open("StopWords.txt", "r") as f:
    stop_words = f.read().splitlines()

for file in files:
    with open(file,'r') as f:
        text = f.read()

def remove_tags(text):
    # Removes the HTML tags from the text
    soup = BeautifulSoup(text, "html.parser")
    strip= soup.get_text(separator=" ")
    return strip

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text) # regex command to remove punctuation

def remove_stopwords(text):
    # Removes stopwords from text
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if not word in stop_words]
    return filtered



# def remove_numbers(text):
#     retru
