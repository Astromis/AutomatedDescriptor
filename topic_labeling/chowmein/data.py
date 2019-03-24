import os
import nltk
import itertools
import codecs
from toolz.functoolz import compose
import cPickle as pickle

CURDIR = os.path.dirname(os.path.realpath(__file__))


def load_line_corpus(path, tokenize=True, merge_pharagraphs=1):
    # path: str Path to corpus
    # tokenize: weather do a tokenization paragraphs into words
    # merge_pharagraphs: how many paragraphs should merge into one (for testing lableling) 
    docs = []
    tmp = []
    with codecs.open(path, "r", "utf8") as f:
        for l in f:
            tmp.append(l.strip())
            
            if len(tmp) != merge_pharagraphs:
                continue
            temp = ' '.join(tmp)
            if tokenize:
                sents = nltk.sent_tokenize(temp.strip().lower())
                docs.append(list(itertools.chain(*map(
                    nltk.word_tokenize, sents))))
            else:
                docs.append(temp.strip())
            tmp = []
        # edge condition handler - if number of paragraphs is not multiple to merge counter
        if len(tmp) != 0:
            temp = ' '.join(tmp)
            if tokenize:
                sents = nltk.sent_tokenize(temp.strip().lower())
                docs.append(list(itertools.chain(*map(
                        nltk.word_tokenize, sents))))
            else:
                docs.append(temp.strip())
    return docs


def load_nips(years=None, raw=False):
    # load data
    if not years:
        years = xrange(2008, 2015)
    files = ['nips-{}.dat'.format(year)
             for year in years]

    docs = []
    for f in files:
        docs += load_line_corpus('{}/datasets/{}'.format(CURDIR, f),
                                 tokenize=(not raw))
        
    return docs                


def load_lemur_stopwords():
    with codecs.open(CURDIR + '/datasets/lemur-stopwords.txt', 
                     'r' 'utf8') as f:
        return map(lambda s: s.strip(),
                   f.readlines())
corp = load_line_corpus("../test.txt")
print(len(corp))
