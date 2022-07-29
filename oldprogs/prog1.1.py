# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:22:09 2022

@author: sandeep
"""

import os

path = input("enter folder path : ").strip()
os.chdir(str(os.getcwd())+"\\"+path)

voc_set = set()

def read_text_file(file):
    with open(file,'r') as f:
        return f.read()

for file in os.listdir():
    if file.endswith(".txt"):
        file_path = f"{file}"
        content = read_text_file(file_path).lower()
        words = content.replace(","," ").replace("."," ").split(" ")
        for word in words:
            voc_set.add(word)
        
vocab = sorted(list(voc_set))
print("vocabulary is",vocab)
print("length",len(vocab))