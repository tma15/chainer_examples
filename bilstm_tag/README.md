# BiLSTM Tagger
BiLSTM tagger can be used for a sequential labeling task such as NP chunking.
This directory includes examples of training and testing BiLSTM model implemetened by chainer.

## Sample data
These example use NP chunking dataset of [CoNLL 2000](http://www.cnts.ua.ac.be/conll2000/chunking/).
For runnning example codes, training data and testing data must be downloaded.

```sh
wget http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz
wget http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz
gunzip train.txt.gz test.txt.gz
```
