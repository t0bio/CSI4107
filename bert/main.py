import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import time
from collections import defaultdict
import tensorflow_hub as hub
from preprocess import *

# Load BERT and Universal Sentence Encoder models
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def indexer(d):
    ind = defaultdict(set)
    for docId, doc in enumerate(d):
        for word in doc:
            ind[word].add(docId)
    return ind

def createDocumentVectors(documents, model, tokenizer=None):
    docVectors = {}
    for docId, doc in documents.items():
        if tokenizer:  # If a tokenizer is provided, use it to tokenize the document
            inputs = tokenizer(doc, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs)
            docVectors[docId] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        else:  # If no tokenizer is provided, assume the model can handle raw text
            docVectors[docId] = model([doc])[0]
    return docVectors

def calculateQueryVector(query, model, tokenizer=None):
    if tokenizer:  # If a tokenizer is provided, use it to tokenize the query
        inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    else:  # If no tokenizer is provided, assume the model can handle raw text
        return model([query])[0]

def getrelevantdocs(query, index):
    docs = set()
    for word in query:
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
            queries = calculateQueryVector(cleaned, bert_model, bert_tokenizer)
            return queries
        
def main():
    documents = loadDocuments()
    index = indexer(documents.values())
    # Create document vectors using both BERT and Universal Sentence Encoder
    bert_docVectors = createDocumentVectors(documents, bert_model, bert_tokenizer)
    use_docVectors = createDocumentVectors(documents, use_model)
    for file in os.listdir('queries'):
        with open(os.path.join('queries', file), 'r') as f:
            text = f.read()
            cleaned = clean(text)
            # Calculate query vectors using both BERT and Universal Sentence Encoder
            bert_query = calculateQueryVector(cleaned, bert_model, bert_tokenizer)
            use_query = calculateQueryVector(cleaned, use_model)
            relevantDocs = getrelevantdocs(text, index)
            # Calculate similarity scores using both BERT and Universal Sentence Encoder
            bert_similarity = cosine_similarity(bert_query, [bert_docVectors[i] for i in relevantDocs])
            use_similarity = cosine_similarity(use_query, [use_docVectors[i] for i in relevantDocs])
            # write the scores out to two different files for each model
            with open(f'bert_results', 'w') as f:
                for i, score in zip(relevantDocs, bert_similarity):
                    f.write(f'{i} {score}\n')
            with open(f'use_results', 'w') as f:
                for i, score in zip(relevantDocs, use_similarity):
                    f.write(f'{i} {score}\n')


def timer():
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end-start} seconds")
    
timer()