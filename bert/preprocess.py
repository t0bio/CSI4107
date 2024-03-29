import nltk
import pandas as pd
import pickle as pk
import numpy as np
import os
import string
import json
import chardet 

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# nltk.download('stopwords')
nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')

# TODO: Reimplement functions here to disable tokenization, since BERT and USE dont take tokenized input

def readFiles(path):
    v = dict()
    for file in os.listdir(path):
        rawdata = open(os.path.join(path, file), "rb").read()
        result = chardet.detect(rawdata)
        charenc = result['encoding']
        with open(os.path.join(path, file), 'r', encoding=charenc) as f:
            parse = BeautifulSoup(f, 'html.parser')
            content = parse.findAll("text")
            tokensFromDoc = []
            for word in content:
                tokensFromDoc.extend(clean(str(word).replace("<text>", "").replace("</text>", "").replace(",", " ").replace("-", " ")))
            uniqueTokensFromDoc = list(dict.fromkeys(tokensFromDoc))
            v[file]=uniqueTokensFromDoc

    jsonFile(v)

    return v


def clean(words):
    stemmer = PorterStemmer()
    stop_words = set(line.strip('\n') for line in open("./StopWords.txt", "r"))

    tokens = []
    for x in word_tokenize(words):
        if x.lower() not in stop_words and x not in string.punctuation and not x.isnumeric():
            # Apply stemming and lowercase conversion
            stemmed_token = stemmer.stem(x.lower())
            # Add the stemmed token to the list
            tokens.append(stemmed_token)
    return tokens

import json

def jsonFile(dictionary):
    with open("tokens.json", "w") as outfile:
        # Convert the dictionary to JSON string
        json_str = json.dumps(dictionary)
        # Write the JSON string to the file
        outfile.write(json_str)


    
    