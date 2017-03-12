# -*- coding: utf-8 -*-

def read_conll_data(vocab_x, vocab_y, data_file, update):
    with open(data_file, 'r') as f:
        x = []
        y = []

        word_ids = []
        tags = []
        for line in f.readlines():
            sp = line.strip().split()
            if len(sp) != 3:
                x.append(word_ids)
                y.append(tags)
                word_ids = []
                tags = []
                continue

            word = sp[0]
            if not vocab_x.has(word):
                if update:
                    vocab_x.add(word)

            if vocab_x.has(word):
                word_ids.append(vocab_x.stoi[word])
            else:
                word_ids.append(vocab_x.stoi["<unk>"])

            tag = sp[2]
            if not vocab_y.has(tag):
                if update:
                    vocab_y.add(tag)

            if vocab_y.has(tag):
                tags.append(vocab_y.stoi[tag])
            else:
                tags.append(vocab_y.stoi["<unk_tag>"])

    return x, y

