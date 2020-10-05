
import numpy as np
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

class GloveEncoder():
    def __init__(self, glove_path, word_dim, max_len):
        self.word_dim = word_dim
        self.max_len = max_len
        self.lm = WordNetLemmatizer()
        glove_path = glove_path+'/glove.6B.'+str(word_dim)+'d.txt'
        self.glove_map = dict(map(self.split,open(glove_path).readlines()))

    def split(self,line):
            p = line.split(' ')
            return (p[0].lower(),np.array([float(f) for f in p[1:]]))

    def encode_phrase(self,phrase):
            em = np.array([])
            for w in word_tokenize(phrase)[:self.max_len]:
                    w = self.lm.lemmatize(w.lower())
                    if w in self.glove_map:
                            if em.size == 0:
                                    em = np.reshape(self.glove_map[w],(1,self.word_dim))
                            else:
                                    em = np.vstack((em,(np.reshape(self.glove_map[w],(1,self.word_dim)))))
            if em.size == 0:
                    return None
            while em.shape[0] < self.max_len:
                    em = np.vstack((em,np.zeros(shape=(1,self.word_dim))))
            return em
