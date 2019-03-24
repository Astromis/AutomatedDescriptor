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

#K = int(sys.argv[1])
#infile = sys.argv[2]



punctuation_pat = re.compile(r"""([!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~])""")
hyphenline_pat = re.compile(r"-\s*\n\s*")
multiwhite_pat = re.compile(r"\s+")
cid_pat = re.compile(r"\(cid:\d+\)")
nonlet = re.compile(r"([^A-Za-z0-9 ])")
def clean_text(txt):
    txt = txt.decode("utf-8")

    txt = txt.lower()
    txt = cid_pat.sub(" UNK ", txt)
    txt = hyphenline_pat.sub("", txt)
    # print punctuation_pat.findall(txt)
    txt = punctuation_pat.sub(r" \1 ", txt)
    txt = re.sub("\n"," NL ", txt)
    txt = nonlet.sub(r" \1 ", txt)

    # txt = punctuation_pat.sub(r"", txt)
    # txt = nonlet.sub(r"", txt)

    txt = multiwhite_pat.sub(" ", txt)
    txt = txt.encode('utf-8')
    return "".join(["START ", txt.strip(), " END"])

#with open(infile,"r") as f:
#    txt = f.read()
#txt = txt.lower().split()

#vecs = np.load("/home/aaa244/storage/arxiv_glove/bigrun/data/mats/vecs.npy")
#words = np.load("/home/aaa244/storage/arxiv_glove/bigrun/data/mats/vocab.npy")

#kv = tools_.load_embeddings("../glove.6B.50d.txt")

#word_lookup = {w:c for c,w in enumerate(kv.vocab.keys()) }

#print "article length:", len(txt)

# word level
#X = []

#mapper = {}
#count = 0
#oov = []
#for i,word in enumerate(txt):
#    if word in word_lookup:
#        mapper[i] = count
#        count += 1
#        X.append( kv[word] )
#    else:
#        oov.append(word)
#open('oov.txt', 'w').write(str(oov))
#mapperr = { v:k for k,v in mapper.iteritems() }

# sen

    


