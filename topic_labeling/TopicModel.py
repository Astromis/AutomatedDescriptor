from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from chowmein.data import load_line_corpus


class TopicModelBase:

    def __init__(self,):
        # topic_words: list of topics that is represented as vector of word values, 
        # that after sorting can represent the most suitible words for describind (ambigious, I know) 
        # this attr must be filled with arrays of topics in the end of fit method 
        self.topic_word_ 
        self.model
    
    def fit(self,doc2word_matrix):
        pass
    
    def print_top_words(self, word_index, n_top_words):
        # word_index: list in wich index of words corresponds to index of matrx doc2word
        for topic_idx, topic in enumerate(self.topic_word_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([word_index[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()

class LDATopicModel(TopicModelBase):
    
    def __init__(self, n_topics=5, max_inter=10, learning_method='online', learning_offset = 50., random_state=0):
        self.model = LatentDirichletAllocation(n_components=n_topics, max_iter=max_inter,
                                learning_method=learning_method,
                                learning_offset=learning_offset,
                                random_state=random_state)
    def fit(self, doc2word_matrix):
        self.model.fit(doc2word_matrix)
        self.topic_word_ = self.model.components_
        
class NMFTopicModel(TopicModelBase):

    def __init__(self, n_topics=5, random_state=1,
                 alpha=.1, l1_ratio=.5):
        self.model = NMF(n_components=n_topics, random_state=random_state,
                         alpha=alpha, l1_ratio=l1_ratio)
                         
    def fit(self, doc2word_matrix):
        self.model.fit(doc2word_matrix)
        self.topic_word_ = self.model.components_
        
class LSATopicModel(TopicModelBase):
    
    def __init__(self, n_topics=5, n_iter=7, random_state=42):
        self.model = TruncatedSVD(n_components=n_topics, n_iter=n_iter, random_state=random_state)
    
    def fit(self, doc2word_matrix):
        self.model.fit(doc2word_matrix)
        self.topic_word_ = self.model.components_
        

