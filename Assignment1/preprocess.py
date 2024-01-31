import nltk
import pandas as pd
import numpy as np
import re
import os
import warnings

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Reading in the documents for preprocessing from the coll folder
cwd = os.getcwd()
path = cwd + "/coll"
files = os.listdir(path)
os.chdir(path)
for file in files:
    with open(file,'r') as f:
        text = f.read()
        

# def remove_tags(text):
#     TAG_RE = re.compile(r'<[^>]+>')
#     return TAG_RE.sub('', text)

# def remove_punctuation(text):
#     return text.translate(str.maketrans('', '', string.punctuation))

# def remove_numbers(text):
#     retru
