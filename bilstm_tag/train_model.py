# -*- coding: utf-8 -*-
from model import BiLSTMTagger
from model import Vocabulary
import corpus

import numpy as np

from chainer import optimizers as O
from chainer import functions as F
from chainer import Variable
from chainer import serializers

def main():
    vocab_x = Vocabulary()
    vocab_x.add("<unk>")

    vocab_y = Vocabulary()
    vocab_y.add("<unk_tag>")

    train_data = "train.txt"
    x, y = corpus.read_conll_data(vocab_x, vocab_y, train_data, True)

    vocab_x.save("vocab_x")
    vocab_y.save("vocab_y")

    tagger = BiLSTMTagger(vocab_x.size(), vocab_y.size())

    optimizer = O.SGD(0.01)
    optimizer.setup(tagger)

    num_data = len(x)
    num_epoch = 10

    for e in range(num_epoch):
        total_loss = 0
        for k, (word_ids, tags) in enumerate(zip(x, y)):
            if k % 1000 == 0:
                print('{}/{}'.format(k, num_data))

            vars_x = [Variable(np.array([wid], dtype=np.int32)) for wid in word_ids]
            vars_y = [Variable(np.array([tid], dtype=np.int32)) for tid in tags]

            ys = tagger(vars_x)
            accum_loss = 0
            for i, y_pred in enumerate(ys):
                loss = F.softmax_cross_entropy(y_pred, vars_y[i])
                accum_loss += loss
            total_loss += accum_loss.data

            tagger.cleargrads()
            accum_loss.backward()
            optimizer.update()
        print('e:{0} {1:.2f}'.format(e, total_loss))
        serializers.save_npz("tagger_e{}".format(e), tagger)
            
    serializers.save_npz("tagger_e{}".format(e), tagger)

if __name__ == '__main__':
    main()
