import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import time
from preprocess import *
from collections import defaultdict


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def index(d):
    ind = defaultdict(set)
    for docId, doc in enumerate(d):
        words = nltk.word_tokenize(doc)
        for word in words:
            ind[word].add(docId)
    return ind

def createDocumentVectors(documents):
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def calculateQueryVector(query):
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def getrelevantdocs(query, index):
    words = nltk.word_tokenize(query)
    docs = set()
    for word in words:
        if word in index:
            docs = docs.union(index[word])
    return docs

def loadDocuments():
    with open('tokens.json', 'r') as f:
        return json.load(f)
    
def loadQueries():
    for file in os.listdir('queries'):
        with open(os.path.join('queries', file), 'r') as f:
            text = f.read()
            cleaned = clean(text)
            queries = calculateQueryVector(cleaned)
            return queries
        
def main():
    documents = loadDocuments()
    index = index(documents.values())
    docVectors = createDocumentVectors(list(documents.values()))
    for file in os.listdir('queries'):
        with open(os.path.join('queries', file), 'r') as f:
            text = f.read()
            cleaned = clean(text)
            query = calculateQueryVector(cleaned)
            relevantDocs = getrelevantdocs(text, index)
            relevantDocVectors = [docVectors[i] for i in relevantDocs]
            similarity = cosine_similarity(query, relevantDocVectors)
            print(similarity)
    queryVector = calculateQueryVector(query)
    relevantDocs = getrelevantdocs(query, index)
    relevantDocVectors = [docVectors[i] for i in relevantDocs]
    similarity = cosine_similarity(queryVector, relevantDocVectors)
    print(similarity)
    
    

def timer():
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end-start} seconds")
    
timer()
    
    
    
        
