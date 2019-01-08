from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import tools
import tools_
import splitters
import representations
import re
import sys

class TextSegmentator:
    def __init__(self, k, level, embeddings_path):
        if level not in ["sentences", "words"]:
            raise ValueError("Unknown level")
        self.k = k
        self.level = level
        self.embedds = tools_.load_embeddings(embeddings_path)
        self.word_lookup = {w:c for c,w in enumerate(self.embedds.vocab.keys()) }
        self.X = []
        self.mapper = {}
        
    def map_to_embeddings(self, text):
        count = 0
        oov = []
        if self.level == "words":
            for i,word in enumerate(text):
                if word in self.word_lookup:
                    self.mapper[i] = count
                    count += 1
                    self.X.append( self.embedds[word] )
                else:
                    oov.append(word)
            if len(oov) == 0:
                print('There are no OOV words')
            else:
                print("Presentage of OOV: %f" % (len(oov) / len(text)  * 100 ))
                open('oov.txt', 'w').write(str(oov))
            
            self.mapperr = { v:k for k,v in self.mapper.iteritems() }
            self.X = np.array(self.X)
            
        elif self.level == "sentences":
            x_sent = []
            for i, sent in enumerate(text):
                self.mapper[i] = count
                count += 1
                for j, word in enumerate(sent):
                    if word in self.word_lookup:
                        x_sent.append(self.embedds[word])
                    else:
                        oov.append(word)    
                self.X.append( np.sum(np.array(x_sent),axis=0) )
            self.mapperr = { v:k for k,v in self.mapper.iteritems() }
            self.X = np.array(self.X)
            print(self.X.shape)

    
    def data_preparing(self, path):
        #text must be a list of words only for word level and list of sentence for sentence level
        
        if self.level == "words":
            preparer = tools_.TextProcessing(level='t')
        else:
            preparer = tools_.TextProcessing(level='st')
        return preparer.proccess(path)
        
    def process(self, path):
        text = self.data_preparing(path)
        self.map_to_embeddings(text)
        sig = splitters.gensig_model(self.X)
        splits,e = splitters.greedysplit(self.X.shape[0], self.k, sig)
        splitsr = splitters.refine(splits, sig, 20)
        with open("results/result{}.txt".format(self.k),"w") as f:
            prev = 0
            for s in splitsr:
                k1 = self.mapperr.get(s,len(text))
                if self.level == "words":
                    f.write(" ".join(text[prev:k1]).replace("NL","\n"))
                elif self.level == "sentences":
                    f.write(". ".join(map(lambda x: ' '.join(x), text[prev:k1]))    .replace("NL","\n"))
                f.write("\nBREAK\n")
                prev = k1


txts = TextSegmentator(5, 'sentences', "../glove.6B.100d.txt")
txts.process("test1.txt")