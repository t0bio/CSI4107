from collections import defaultdict
import json
import time
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

# Load Universal Sentence Encoder
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def indexer(d):
    ind = defaultdict(set)
    for docId, doc in d.items():
        for word in doc:
            ind[word].add(docId)
    return ind

def getrelevantdocs(query, index):
    docs = set()
    for word in query:
        if word in index:
            docs = docs.union(index[word])
    if not docs:
        docs = set(index.keys())
    return docs

def loadDocuments():
    with open('tokens.json', 'r') as f:
        return json.load(f)
    
def loadQueries():
    with open('queries.json', 'r') as f:
        return json.load(f)
        

# def createDocumentVectors(documents, model=None, use_model=None):
#     docVectors = {}
#     for docId, doc in documents.items():
#         doc = ' '.join(doc)
#         if use_model:  # If the USE model is provided, use it to create embeddings
#             docVectors[docId] = use_model([doc]).numpy()[0]
#     return docVectors

def createDocumentVectors(documents):
    docVectors = {}
    for docId, doc in documents.items():
        doc_text = ' '.join(doc)
        # Encode the document using USE
        embeddings = use_model([doc_text])
        docVectors[docId] = embeddings.numpy().squeeze()
    return docVectors

# def calculateQueryVector(query, model=None, use_model=None):
#     query = ' '.join(query)
#     if use_model:  # If the USE model is provided, use it to create embeddings
#         return use_model([query]).numpy()[0]

def calculateQueryVector(query):
    query = ' '.join(query)
    embeddings = use_model([query])
    return embeddings.numpy().squeeze()

def main():
    uresults = dict()
    documents = loadDocuments()
    queries = loadQueries()
    index = indexer(documents)
    # Create document vectors using Universal Sentence Encoder
    # use_docVectors = createDocumentVectors(documents, use_model=use_model)
    use_docVectors = createDocumentVectors(documents)

    #processing each query:
    for queryId, query in queries.items():
        # Calculate query vectors using Universal Sentence Encoder
        # use_queryVector = calculateQueryVector(' '.join(query), use_model=use_model)
        use_queryVector = calculateQueryVector(' '.join(query))
        
        # Get relevant documents using the index
        relevantDocs = getrelevantdocs(query, index)
        
        # Calculate cosine similarity between query and document vectors
        usesimilarity = {docId: cosine_similarity(use_queryVector.reshape(1,-1), docVector.reshape(1,-1))[0][0] for docId, docVector in use_docVectors.items() if docId in relevantDocs}
        #sort by similarity
        utopdocs = sorted(usesimilarity, key=usesimilarity.get, reverse=True)[:1000]
        #store results
        uresults[queryId] = utopdocs
        
    # writing the results in trec_eval format
    with open('use_results.txt', 'w') as f:
        for queryId, docs in uresults.items():
            for rank, docId in enumerate(docs):
                f.write(f"{queryId} Q0 {docId} {rank+1} {1.0} use\n")
                
def timer():
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end-start} seconds")
    
timer()