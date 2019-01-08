from gensim.models import KeyedVectors
from os import listdir
import codecs
import itertools

def load_embeddings(path):
    """
    Loads pre-trained glove word embeddings with gensim

    input: path to pretrained word embeddings.
            it must be in the glove format. can be obtainable from official site
    output: gensim's object of KeyedVectors
    """
    tmp = 0
    print("Import word embeddings.")
    tmp_name = 'tmp_' + path[path.rfind('/')+1:] + '.data'
    if tmp_name not in listdir('.'):
        print("Required format not found. Translating") 
        tmp = open(tmp_name, 'w')
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove2word2vec(path, tmp)
        tmp = open(tmp_name, 'r')
    else:
        tmp = open(tmp_name, 'r')

    return KeyedVectors.load_word2vec_format(tmp)


def default_remove_punct(tokens):
    return [word for word in tokens if word.isalpha()]

def default_remove_punct2(tokens):
    return [word for word in tokens if word not in u'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~']


class TextProcessing:

    def __init__(self, level='ps', remove_sw=True, remove_punct=default_remove_punct2):
        if level not in ['pst','ps','st','t','s','p', 'pt']:
            raise ValueError("Unknown level")
        self.level = level
        self.remove_sw = remove_sw
        self.remove_punct = remove_punct
    
    def proccess(self, path):
        """
        main function that proceeds proccess information
        level - look of the result at the end
            pst - list of paragraphs, containing list of sentences with lists of words
            ps - list of paragraphs with list of sentences
            pt - list of paragraphs with lists of words
            st - list of sentnces with lists of words
            t - list of words

        remove_sw - wether remove stopwords
        remove_punct - function that defines a process of punctuation removing 

        input - path to text
        output - processed text, look of which is defined by level
        """
        text = []
        with codecs.open(path, "r", "utf8") as f:
            for line in f:
                line = line.lower()
                if self.level == 'pst' or self.level == 'st':
                    text.append(self.split_into_senttok(line))
                elif self.level == 'ps':
                    text.append(self.split_into_sencetce(line))
                elif self.level == 'pt' or self.level == 't':
                    text.append(self.split_into_tokens(line))
                else:
                    raise NotImplementedError
            if self.level == 'st':
                text = [x for x in itertools.chain(*text)]
            elif self.level == 't':
                text = [x for x in itertools.chain(*text)]
        return text


    def split_into_sencetce(self, text):
        """
        Tokenizes text into list of sentences

        input: text, str
        output: tokenized sentences, list of str
        """
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)

    def split_into_tokens(self, text, remove_sw=True, remove_punct=default_remove_punct2):
        """
        Tokenizes text into list of words

        input: 
                text, str
                remove_sw, bool wether need to delete stopwords
        output: tokenized words, list of str
        """
        import nltk
        word_tokens = nltk.word_tokenize(text)
        if remove_punct != None:
            word_tokens = remove_punct(word_tokens)
        if remove_sw:
            from nltk.corpus import stopwords 
            stop_words = set(stopwords.words('english')) 
            word_tokens = [w for w in word_tokens if not w in stop_words]
        
        return word_tokens

    def split_into_senttok(self, text):
        """
        Tokenzes text into sentences, and sentences into words.

        input: text, sts
        output: tokenized lists(sentences) of words, list of lists of words
        """
        out = []
        for sent in self.split_into_sencetce(text):
            out.append(self.split_into_tokens(sent))
        return out


#test = "At eight o'clock on Thursday morning... Arthur didnt feel very good. But after he ate, he did feeling better."
#print(split_into_senttok(test))

#text_peprocessing("text")
from nltk.collocations import BigramCollocationFinder
from nltk import BigramAssocMeasures

measures = BigramAssocMeasures()
tp = TextProcessing(level='st')
data = tp.proccess('test.txt')
finder = BigramCollocationFinder.from_documents(data)
bigrams = finder.nbest(measures.fisher,10)
print(bigrams)