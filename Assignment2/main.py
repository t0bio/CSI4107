import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import time
from collections import defaultdict
# from preprocess import *
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow_hub as hub

# Load BERT and Universal Sentence Encoder models
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def indexer(d):
    ind = defaultdict(set)
    for docId, doc in d.items():
        for word in doc:
            ind[word].add(docId)
    return ind

def createDocumentVectors(documents, model, tokenizer=None):
    docVectors = {}
    for docId, doc in documents.items():
        doc = ' '.join(doc)
        inputs = tokenizer(doc, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        docVectors[docId] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return docVectors

def createDocumentVectorsUSE(documents):
    docVectors = {}
    for docId, doc in documents.items():
        doc_text = ' '.join(doc)
        # Encode the document using USE
        embeddings = use_model([doc_text])
        docVectors[docId] = embeddings.numpy().squeeze()
    return docVectors

def calculateQueryVector(query, model, tokenizer=None):
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def calculateQueryVectorUSE(query):
    query = ' '.join(query)
    embeddings = use_model([query])
    return embeddings.numpy().squeeze()

def getrelevantdocs(query, index):
    docs = set()
    for word in query:
        if word in index:
            if not docs:
                docs = index[word]
            else:
                docs = docs.intersection(index[word])
    return docs

def loadDocuments():
    with open('tokens.json', 'r') as f:
        return json.load(f)
    
def loadQueries():
    with open('queries.json', 'r') as f:
        return json.load(f)
        
def main():
    bresults = dict()
    uresults = dict()
    # useresults = dict()
    documents = loadDocuments()
    queries = loadQueries()
    index = indexer(documents)
    # print (index)
    # Create document vectors using both BERT and Universal Sentence Encoder
    bert_docVectors = createDocumentVectors(documents, bert_model, bert_tokenizer)
    use_docVectors = createDocumentVectorsUSE(documents)

    # #processing each query:
    # for queryId, query in queries.items():
    #     # Calculate query vectors using both BERT and Universal Sentence Encoder
    #     bert_queryVector = calculateQueryVector(' '.join(query), bert_model, bert_tokenizer)
    #     # use_queryVector = calculateQueryVector(' '.join(query), use_model)
        
    #     # Get relevant documents using the index
    #     relevantDocs = getrelevantdocs(query, index)
        
    #     # Calculate cosine similarity between query and document vectors
    #     # bertsimilarity = {docId: cosine_similarity(bert_queryVector.reshape(1.-1).reshape(1,-1))[0][0] for docId, docVector in bert_docVectors.items() if docId in relevantDocs}
    #     bertsimilarity = {docId: cosine_similarity(bert_queryVector.reshape(1,-1), docVector.reshape(1,-1))[0][0] for docId, docVector in bert_docVectors.items() if docId in relevantDocs}
    #     #sort by similarity
    #     btopdocs = sorted(bertsimilarity, key=bertsimilarity.get, reverse=True)[:1000]
    #     #store results
    #     bresults[queryId] = btopdocs
        
    # # writing the results in trec_eval format
    # with open('bert_results.txt', 'w') as f:
    #     for queryId, docs in bresults.items():
    #         for rank, docId in enumerate(docs):
    #             f.write(f"{queryId} Q0 {docId} {rank+1} {1.0} bert\n")
    # Open the file before processing the queries
    with open('bert_results.txt', 'w') as f:
        for queryId, query in queries.items():
            querys = ' '.join(query)
            bert_queryVector = calculateQueryVector(' '.join(query), bert_model, bert_tokenizer)
            relevantDocs = getrelevantdocs(querys, index)
            print (relevantDocs)
            bertsimilarity = {docId: cosine_similarity(bert_queryVector.reshape(1,-1), docVector.reshape(1,-1))[0][0] for docId, docVector in bert_docVectors.items() if docId in relevantDocs}
            btopdocs = sorted(bertsimilarity, key=bertsimilarity.get, reverse=True)[:1000]
            bresults[queryId] = btopdocs

            # Write the results for the current query to the file
            for rank, docId in enumerate(btopdocs):
                f.write(f"{queryId} Q0 {docId} {rank+1} {1.0} bert\n")
                
    

def timer():
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end-start} seconds")
    
timer()