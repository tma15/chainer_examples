# -*- coding: utf-8 -*-
import sys

import numpy as np

import chainer
from chainer import serializers
from chainer import Variable

import corpus
from model import BiLSTMTagger
from model import load_vocab

def main():
    vocab_x = load_vocab("vocab_x")
    vocab_y = load_vocab("vocab_y")

    x, y = corpus.read_conll_data(vocab_x, vocab_y, "test.txt", False)
#    x, y = corpus.read_conll_data(vocab_x, vocab_y, "train.txt", False)

    tagger = BiLSTMTagger(vocab_x.size(), vocab_y.size())
    serializers.load_npz(sys.argv[1], tagger)

    for k, (word_ids, tags) in enumerate(zip(x, y)):

        vars_x = [Variable(np.array([wid], dtype=np.int32)) for wid in word_ids]
        vars_y = [Variable(np.array([tid], dtype=np.int32)) for tid in tags]

        ys = tagger(vars_x)
        for i in range(len(ys)):
            if i >= len(word_ids) or i >= len(tags):
                sys.stderr.write('i: {} {} {}\n'.format(i, len(word_ids), len(tags)))
                continue
            word = vocab_x.itos[word_ids[i]]

            if tags[i] >= vocab_y.size():
                sys.stderr.write('{}\n'.format(tags[i]))
                continue
            tag = vocab_y.itos[tags[i]]

            scores = ys[i].data[0]
            max_id = np.argmax(scores)
            if max_id >= vocab_y.size():
                continue
            pred_tag = vocab_y.itos[max_id]

            print('{} POS {} {}'.format(word, tag, pred_tag))
        print(flush=True)


if __name__ == '__main__':
    main()
