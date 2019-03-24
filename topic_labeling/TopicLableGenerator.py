import argparse
import lda
from gensim.models import ldamodel
from gensim.corpora.dictionary import Dictionary
from gensim.models.lsimodel import LsiModel


import numpy as np

from sklearn.feature_extraction.text import (CountVectorizer
                                             as WordCountVectorizer)
from chowmein.text import LabelCountVectorizer
from chowmein.label_finder import BigramLabelFinder
from chowmein.label_ranker import LabelRanker
from chowmein.pmi import PMICalculator
from chowmein.corpus_processor import (CorpusWordLengthFilter,
                                       CorpusPOSTagger,
                                       CorpusStemmer)
from chowmein.data import (load_line_corpus, load_lemur_stopwords)

class TopicLableGenerator():
    """
    Class for generating labels for topics in topic models.

    corpus_path(str): path to corpus file, where each line is being seen as one document.
                        (So corpus may be a list of string)
    preprocessing_steps(list of [wordlen,tag,stem]): definition of preprocessing steps
    label_min_df(int): Minimum document frequency for label (It is inherited from sklearn)
    tabel_tags(list of str): POS tags, by wich filtering will be performed.
    n_lables(int): number of lables per topic
    --------------------------
    LDA params(should be excluded from this class)
    n_topics(int): number of topics
    lda_random_state(int): random state
    ;da_n_iter(int): number of iterations

    Gereal use:
    from TopicLableGenerator import *

    lables_gen = TopicLableGenerator('./chowmein/datasets/nips-2014.dat', ['wordlen', 'tag'])

    lables_gen.execute(100,8)
    lables_gen.print_topical_lables
    """
    def __init__(self,corpus_path, preprocessing_steps,
                     topic_model,
                     label_min_df=1,
                     label_tags=['NN,NN', 'JJ,NN'], n_labels=8,
                     ):
        
        self.tag_constraints = []

        self.preprocessing_steps = preprocessing_steps # this is a list

        self._preprocessing(corpus_path, label_tags)
        
        self.finder = BigramLabelFinder('pmi', min_freq=label_min_df,
                        pos=self.tag_constraints)

        self.pmi_cal = PMICalculator(
                        doc2word_vectorizer=WordCountVectorizer(
                        min_df=2,
                        stop_words=load_lemur_stopwords()),
                        doc2label_vectorizer=LabelCountVectorizer())

        self.model = topic_model
        
        self.ranker = LabelRanker(apply_intra_topic_coverage=False)


    def _preprocessing(self, corpus_path, label_tags):

        self.docs = load_line_corpus(corpus_path)
        if 'wordlen' in self.preprocessing_steps:
            print("Word length filtering...")
            wl_filter = CorpusWordLengthFilter(minlen=3)
            self.docs = wl_filter.transform(self.docs)

        if 'stem' in self.preprocessing_steps:
            print("Stemming...")
            stemmer = CorpusStemmer()
            self.docs = stemmer.transform(self.docs)

        if 'tag' in self.preprocessing_steps:
            print("POS tagging...")
            tagger = CorpusPOSTagger()
            self.tagged_docs = tagger.transform(self.docs)

        if label_tags != ['None']:
            for tags in label_tags:
                self.tag_constraints.append(tuple(map(lambda t: t.strip(),
                                                tags.split(','))))

        if len(self.tag_constraints) == 0:
            self.tag_constraints = None

        print("Tag constraints: {}".format(self.tag_constraints))

    def execute(self, n_cand_labels, n_labels):

        print("Generate candidate bigram labels(with POS filtering)...")
        if self.tag_constraints:
            assert 'tag' in self.preprocessing_steps, \
            'If tag constraint is applied, pos tagging(tag) should be performed'
            cand_labels = self.finder.find(self.tagged_docs, top_n=n_cand_labels)
        else:  # if no constraint, then use untagged docs
            cand_labels = self.finder.find(self.docs, top_n=n_cand_labels)

        print("Collected {} candidate labels".format(len(cand_labels)))

        print("Calculate the PMI scores...")
        self.pmi_w2l = self.pmi_cal.from_texts(self.docs, cand_labels)

        print("Topic modeling using LDA...")
        self.model.fit(self.pmi_cal.d2w_)
      
        self.labels = self.ranker.top_k_labels(topic_models=self.model.topic_word_,
                                    pmi_w2l=self.pmi_w2l,
                                    index2label=self.pmi_cal.index2label_,
                                    label_models=None,
                                    k=n_labels)
        return self.labels
    
    @property
    def print_topical_lables(self):
        print("\nTopical labels:")
        print("-" * 20)
        for i, labels in enumerate(self.labels):
            print(u"Topic {}: {}\n".format(
                i,
                ', '.join(map(lambda l: ' '.join(l), labels))
            ))

    @property
    def print_topical_words(self, n_top_words):
        print("\nTopical words:")
        print("-" * 20)
        for i, topic_dist in enumerate(self.model.topic_word_):
            top_word_ids = np.argsort(topic_dist)[:-n_top_words:-1]
            topic_words = [self.pmi_cal.index2word_[id_]
                        for id_ in top_word_ids]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))
