"""
A set of useful tools for handing the text segmentation tasks.
"""
import numpy as np
from itertools import chain, tee#, izip
# from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import KeyedVectors

stopword_set = set()

with open("STOPWORD.list") as f:
    for line in f:
        stopword_set.add(line.strip())
# add punctuation
stopword_set.update(["''",",",".","``","'","!",'"',"#","$","%","&","(",")","*","+","-","/",
    ":",";","<","=",">","?","@","[","\\","]","^","_","`","{","|","}","~"])

stemmer = PorterStemmer()

# Utility

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)# izip


def seg_iter(splits):
    return pairwise([0] + splits)
    
def load_embeddings(path, type):
    """
    Loads pre-trained glove word embeddings with gensim

    input: path to pretrained word embeddings.
            it must be in the glove format. can be obtainable from official site
    output: gensim's object of KeyedVectors
    """
    if type == 'glove':
        tmp = 0
        print("Import glove word embeddings.")
        tmp_name = 'tmp_' + path[path.rfind('/')+1:] + '.data'
        if tmp_name not in listdir('.'):
            print("Required format not found. Translating...") 
            tmp = open(tmp_name, 'w')
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove2word2vec(path, tmp)
            tmp = open(tmp_name, 'r')
        else:
            tmp = open(tmp_name, 'r')

        return KeyedVectors.load_word2vec_format(tmp)
    elif type == 'fasttext':
        print("Import fasttext word embeddings.")
        return KeyedVectors.load(path)
