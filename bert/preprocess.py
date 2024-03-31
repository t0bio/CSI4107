import nltk
import pandas as pd
import pickle as pk
import numpy as np
import os
import string
import json
import chardet 
import re
import json
import time

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
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
            text = f.read()
            docno = re.search(r'<DOCNO>(.*?)</DOCNO>', text)
            if docno:
                docno = docno.group(1)
            else:
                docno = file
            parse = BeautifulSoup(text, 'html.parser')
            content = parse.findAll("text")
            cleanedTextFromDoc = ""
            for text in content:
                cleanedText = clean(str(text).replace("<text>", "").replace("</text>", "").replace(",", " ").replace("-", " "))
                cleanedTextFromDoc += cleanedText + " "

            cleanedTextFromDoc = cleanedTextFromDoc.strip()
            v[file]=cleanedTextFromDoc

    jsonFile(v)

    return v


# NO TOKENIZATION
def clean(text):
    stemmer = PorterStemmer()
    stop_words = set(line.strip('\n') for line in open("./StopWords.txt", "r"))
    
    #remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    #convert to lowercase
    text = text.lower()
    
    #remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    #remove stopwords, numeric words, apply stemming
    words = text.split()  
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words and not word.isnumeric()]
    
    #reconstruct text
    cleaned_text = ' '.join(cleaned_words)
    
    return cleaned_text

def jsonFile(dictionary):
    with open("tokens.json", "w") as outfile:
        # Convert the dictionary to JSON string
        json_str = json.dumps(dictionary)
        # Write the JSON string to the file
        outfile.write(json_str)


def timer():
    start = time.time()
    readFiles("/Users/vanishabagga/Desktop/a2/CSI4107/Assignment1/coll")
    end = time.time()
    print(f"Time taken: {end-start} seconds")
timer()
    