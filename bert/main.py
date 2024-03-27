import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import time
from preprocess import *


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Using BERT to compute the similarity between the query and every document in the collection
# Use a boolean index to restrict the calculations to only documents that have at least one query word

def index(d):
    tokens = dict()
    for key in d:
        words = d.get(key)
        for word in words:
            if word not in tokens:
                tokens[word] = []
            tokens[word].append(key)
    return tokens

def createDocumentVectors(documents):
    docVec = {}
    for docId, doc in documents.items():
        inputs = tokenizer(doc, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        docVec[docId] = torch.mean(last_hidden_states, dim=1).detach().numpy()
    return docVec

def calculateQueryVector(query):
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return torch.mean(last_hidden_states, dim=1).detach().numpy()

def retrieveAndRank(queryVec, docVec):
    scores = {}
    for docId, doc in docVec.items():
        scores[docId] = cosine_similarity(queryVec, doc).item()
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def print_vocabulary(inverted_index):
    with open('vocabulary.txt', 'w') as file:
        for x, (word, y) in enumerate(inverted_index.items()):
            file.write(f"{word}\n")
            if x == 99:  # Stop after writing the first 100 tokens
                break

def main():
    path = './coll/'
    path2 = './queries/'
    pre = readFiles(path)
    next = index(pre)
    docvec = createDocumentVectors(pre)
    fin = []
    print_vocabulary(next)

    # loop over the files in the queries folder and store in a json
    for file in os.listdir(path2):
        filename, filenametxt = os.path.splitext(file)
        with open(os.path.join(path2, file), 'r') as f:
            text = f.read()
            textdic = clean(text)
            queryvec = calculateQueryVector(textdic)
            results = retrieveAndRank(queryvec, docvec)
            fin.extend([(int(filename), id, rank, score) for rank, (id, score) in enumerate(results[:1000],1)])
            
            fin.sort()

        with open('results.txt', 'a') as out:
            for filename, id, rank, score in fin:
                out.write(f"{filename} Q0 {id} {rank} {score} TestRun\n")

def timer():
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end-start} seconds")
    
timer()
    
    
    
        
