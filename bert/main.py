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
    query = ' '.join(query)
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
        docs = docs.union(index[word])
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
    index = indexer(documents.values())
    # Create document vectors using both BERT and Universal Sentence Encoder
    bert_docVectors = createDocumentVectors(documents, bert_model, bert_tokenizer)
    use_docVectors = createDocumentVectorsUSE(documents)

    #processing each query:
    for queryId, query in queries.items():
        # Calculate query vectors using both BERT and Universal Sentence Encoder
        bert_queryVector = calculateQueryVector(' '.join(query), bert_model, bert_tokenizer)
        use_queryVector = calculateQueryVectorUSE(' '.join(query))
        # use_queryVector = calculateQueryVector(' '.join(query), use_model)

        # Reshape the query vector for cosine similarity calculation
        use_queryVector = use_queryVector.reshape(1, -1)
        
        # Get relevant documents using the index
        relevantDocs = getrelevantdocs(query, index)
        
        # Calculate cosine similarity between query and document vectors
        bertsimilarity = {docId: cosine_similarity(bert_queryVector, docVector)[0][0] for docId, docVector in bert_docVectors.items() if docId in relevantDocs}
        usesimilarity = {docId: cosine_similarity(use_queryVector, docVector.reshape(1, -1))[0][0] for docId, docVector in use_docVectors.items() if docId in relevantDocs}
        #sort by similarity
        btopdocs = sorted(bertsimilarity, key=bertsimilarity.get, reverse=True)[:1000]
        utopdocs = sorted(usesimilarity, key=usesimilarity.get, reverse=True)[:1000]
        #store results
        bresults[queryId] = btopdocs
        uresults[queryId] = utopdocs
        
    # writing to files
    with open('bertresults.json', 'w') as f:
        json.dump(bresults, f)

    with open('useresults.json', 'w') as f:
        json.dump(uresults, f)


def timer():
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end-start} seconds")
    
timer()