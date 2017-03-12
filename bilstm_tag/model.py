# -*- coding: utf-8 -*-
import numpy as np

import chainer
from chainer import links as L
from chainer import functions as F
from chainer import Variable

class Vocabulary:
    def __init__(self):
        self.itos = []
        self.stoi = {}

    def size(self):
        return len(self.itos)

    def has(self, elem):
        return elem in self.stoi

    def add(self, elem):
        i = len(self.itos)
        self.stoi[elem] = i
        self.itos.append(elem)

    def save(self, vocab_file):
        with open(vocab_file, 'w') as f:
            for elem in self.itos:
                print('{}'.format(elem), file=f)

def load_vocab(vocab_file):
    vocab = Vocabulary()
    with open(vocab_file, 'r') as f:
        for line in f.readlines():
            elem = line.strip()
            vocab.add(elem)
    return vocab

class BiLSTMTagger(chainer.Chain):
    def __init__(self, n_vocab, n_class, n_emb=200, n_hid=100):

        super(BiLSTMTagger, self).__init__(
                embed=L.EmbedID(n_vocab, n_emb),

                ### parameters for forward LSTM
                xh=L.Linear(n_emb, 4 * n_hid),
                hh=L.Linear(n_hid, 4 * n_hid),

                ### parameters for backward LSTM
                xh_b=L.Linear(n_emb, 4 * n_hid),
                hh_b=L.Linear(n_hid, 4 * n_hid),

                ho=L.Linear(2 * n_hid, n_hid),
                o=L.Linear(n_hid, n_class),
                )

        mu = 0.
        sigma = 0.05
        for param in self.params():
#            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
            param.data[...] = np.random.normal(mu, sigma, param.data.shape)

        self.n_hid = n_hid

    def __call__(self, words):
        ### forward LSTM
        c = chainer.Variable(np.zeros((1, self.n_hid), dtype=np.float32))
        h = chainer.Variable(np.zeros((1, self.n_hid), dtype=np.float32))

        hs = []
        for word in words:
            e = self.embed(word)
            lstm_in = self.xh(e) + self.hh(h)
            c, h = F.lstm(c, lstm_in)
            hs.append(h)

        ### backward LSTM
        c = chainer.Variable(np.zeros((1, self.n_hid), dtype=np.float32))
        h = chainer.Variable(np.zeros((1, self.n_hid), dtype=np.float32))

        hs_b = []
        for word in reversed(words):
            e = self.embed(word)
            lstm_in = self.xh_b(e) + self.hh_b(h)
            c, h = F.lstm(c, lstm_in)
            hs_b.append(h)
        hs_b.reverse()

        ### MLP
        ys = []
        for h_i, hs_b_i in zip(hs, hs_b):
            o1 = self.ho(F.concat([h_i, hs_b_i]))
            y_i = self.o(F.sigmoid(o1))
            ys.append(y_i)
        return ys

if __name__ == '__main__':
    vocab = Vocabulary()

    words = ["I", "have", "an", "apple"]
    word_ids = []
    for word in words:
        if not vocab.has(word):
            vocab.add(word)
        wid = vocab.stoi[word]
        word_ids.append(wid)

    vars_ = [Variable(np.array([wid], dtype=np.int32)) for wid in word_ids]

    n_vocab = len(set(words))
    n_class = 5
    model = BiLSTMTagger(n_vocab, n_class)

    ys = model(vars_)
