# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:22:09 2022

@author: sandeep
"""

import os, timeit
from math import log, sqrt
from pandas import DataFrame


dnames = []
voc_set = set()
corpus = []

#utility to convert to pandas df
def pr_p(x):
    return DataFrame(x)
    
#utlity to tokenize a long string
def tokenize(s):
    return s.lower().replace(","," ",1000).replace("."," ",1000).replace("  "," ",10000).replace('"',' ',1000).replace('\n',' ',1000
        ).replace("'",' ',1000).replace("/",' ',1000).replace("\\"," ",1000).replace("("," ",1000).replace(";"," ",1000
        ).replace(")"," ",1000).replace("]"," ",1000).replace("["," ",1000).replace("-"," ",100).strip().split(" ")
    
#utility to make frequency dictionary
def freq(l):
	d = {}
	for x in l:
		d[x] = d.get(x,0)+1
	return d

#utility to find norm of a vector
def norm(vec):
	return sqrt(sum([v**2 for v in vec]))

#utility to find cosine similarity between two vectors
def cosine_sim(vec1, vec2, norm1, norm2):
    if norm1==0 or norm2==0: return 0
    dot_product = sum([vec1[i]*vec2[i] for i in range(M)])
    return (dot_product/norm1)/norm2
    
#utility to read text file
def read_text(file):
    with open(file,'r') as f:
        return f.read()

'''changing directory to read corpus'''
path = input("enter folder path : ").strip()
curr_dir = str(os.getcwd())
new_path = ""
if "\\" in curr_dir:
	new_path = curr_dir + "\\" + path
elif "/" in curr_dir:
	new_path = curr_dir + "/" + path
os.chdir(new_path)
	

'''reading the documents in corpus'''
for file in os.listdir():
    if file.endswith(".txt"):
        dnames.append(file.replace('.txt',''))
        file_path = f"{file}"
        content = read_text(file_path)
        words = tokenize(content)
        doc = freq(words)
        voc_set = voc_set.union(doc)
        corpus.append(doc)

vocab = sorted(list(voc_set))
M = len(vocab)
N = len(corpus)

print("\n\nVocabulary is :\n",vocab)
print("\nLength of vocab : ",M)
print("No of documents : ",N)



'''raw frequencies of jth term in ith document'''
raw_freq = []
for word in vocab:
    vec = [doc.get(word,0) for doc in corpus]
    raw_freq.append((word,vec))
    
print("\n\n raw frequencies:\n", pr_p(raw_freq))

'''term frequencies log normalized'''
tf = []
for entry in raw_freq:
    tf.append( (entry[0], [1+log(x,2) if x!=0 else 0 for x in entry[1]]) )
print("\n\n term frequencies log normalized:\n",pr_p(tf))

'''document frequecies of each term'''
df = []
for word in vocab:
    df_i = 0
    for doc in corpus:
        if word in doc:
            df_i+=1
    df.append( df_i )
print("\n\ndf:\n",df)

'''idf vector'''
idf = [ (vocab[i], log(N/df[i] ,2)) for i in range(M) ]
print("\n\nidf vector:\n",pr_p(idf))

'''tdm matrix by tf and idf'''
tdm = [(
            tf[j][0], 
            [tf[j][1][i]*idf[j][1] for i in range(N)]
        ) for j in range(M)]
print("\n\ttdm matrix:\n",pr_p(tdm))

def doc_v(i):
    return [tdm[j][1][i] for j in range(M)]
    
norms = [norm(doc_v(i)) for i in range(N)]
	
''' Cosine Similarity of two documents '''
csm = []
for i in range(N):
    for j in range(i, N):
        csm.append(
                ( dnames[i], dnames[j], 
                  cosine_sim(doc_v(i), doc_v(j), norms[i], norms[j]) )
        )
print("\nCosine Similarity between documents\n",pr_p(csm))




''' Matching Query for search results '''
query = input("\n Enter your query : ")

start_time = timeit.default_timer()
query_t = freq(tokenize(query))
query_v = [query_t[w] if w in query_t else 0 for w in vocab]
cs_r = [ (dnames[i],
       cosine_sim(query_v, doc_v(i), norm(query_v), norms[i])) for i in range(N) ]
cs_r_sorted = sorted( filter(lambda x:x[1]!=0, cs_r) , key = lambda x: x[1], reverse=True)
end_time = timeit.default_timer()
total_time = end_time - start_time

print("     query vector :", query_v)
print("cosine sim. with q:\n", cs_r)
print("\nFetched",len(cs_r_sorted),"results in",total_time,"seconds.")
print(pr_p(cs_r_sorted))
