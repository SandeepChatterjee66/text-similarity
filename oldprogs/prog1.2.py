# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:22:09 2022

@author: sandeep
"""

import os
from math import log, sqrt
from pandas import DataFrame

path = input("enter folder path : ").strip()
curr_dir = str(os.getcwd())
new_path = ""
if "\\" in curr_dir:
	new_path = curr_dir + "\\" + path
elif "/" in curr_dir:
	new_path = curr_dir + "/" + path
	
os.chdir(new_path)

def pr_p(x):
    return DataFrame(x)

def tokenize(s):
    return s.lower().replace(","," ",100).replace("."," ",100).replace("  "," ",1000).strip().split(" ")

def read_text_file(file):
    with open(file,'r') as f:
        return f.read()

voc_set = set()
corpus = []

for file in os.listdir():
    if file.endswith(".txt"):
        file_path = f"{file}"
        content = read_text_file(file_path)
        words = tokenize(content)
        doc = {}
        for word in words:
            voc_set.add(word)
            if word not in doc:
                doc[word]= 1
            else:
                doc[word]+=1
        corpus.append(doc)

vocab = sorted(list(voc_set))
M = len(vocab)
N = len(corpus)

print("\n\nvocabulary is",vocab)
print("\nlength of vocab",M)
print("no of documents",N)


raw_freq = []
for word in vocab:
    vec = []
    for doc in corpus:
        if word not in doc:
            vec.append(0)
        else:
            vec.append(doc[word])
    raw_freq.append((word,vec))
    
print("\n\n raw frequencies:\n", pr_p(raw_freq))


tf = []
for entry in raw_freq:
    tf.append( (entry[0], [1+log(x,2) if x!=0 else 0 for x in entry[1]]) )
print("\n\n term frequencies log normalized:\n",pr_p(tf))


idf = []
for word in vocab:
    df = 0
    for doc in corpus:
        if word in doc:
            df+=1
    idf.append( (word, log(N/df ,2)) )
print("\n\nidf vector:\n",pr_p(idf))

tdm = []
for j in range(M):
    tdm.append( (
            tf[j][0], 
            [tf[j][1][i]*idf[j][1] for i in range(N)]
        ) )
print("\n\ttdm matrix:\n",pr_p(tdm))

def cosine_sim(vec1, vec2):
    dot_product = sum([vec1[i]*vec2[i] for i in range(M)])
    norm1 = sqrt(sum([v**2 for i,v in enumerate(vec1)]))
    norm2 = sqrt(sum([v**2 for i,v in enumerate(vec2)]))
    return (dot_product/norm1)/norm2

def doc_vector(i):
    return [tdm[j][1][i] for j in range(M)]

csm = []
for i in range(N):
    for j in range(i, N):
        csm.append(
                ( "doc"+str(i+1), "doc"+str(j+1), cosine_sim(doc_vector(i), doc_vector(j)) )
        )
print("\n\ncosine similarity between documents\n",pr_p(csm))
        
query = input("enter your query : ")

query_t = tokenize(query)
query_v = [query_t.count(word) for word in vocab]
print("query vector",query_v)

cs_r = [ ("doc"+str(i+1),cosine_sim(query_v, doc_vector(i))) for i in range(N) ]
print("cosine similarity of documents with query:\n", cs_r)

cs_r_sorted = sorted(cs_r , key = lambda x: x[1], reverse=True)
print("\nrank of documents",pr_p(cs_r_sorted))
