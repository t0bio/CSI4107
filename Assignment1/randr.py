# Step 3. [10 points] Retrieval and Ranking:  Use the inverted index (from step 2) to find the limited set of documents that contain at least one of the query words. Compute the cosine similarity scores between a query and each document. 
# •       Input: One query and the Inverted Index (from Step2)
# •       Output: Similarity values between the query and each of the documents. Rank the documents in decreasing order of similarity scores.
import math
from index import index
from index import createDocumentVectors


# query is a preprocessed dict of test query words
# indexDict is the inverted index
# collection is the preprocessed collection of documents

def cosine_similarity(v1, v2):# cosine similarity between two vecs
        sumx = 0
        sumy = 0
        sumxy = 0
        ans = 0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            sumx += x * x
            sumy += y * y
            sumxy += x * y
            ans = sumxy/math.sqrt(sumx * sumy)
            return ans 
        



            
                       