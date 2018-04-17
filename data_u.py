# data.py
import os
# import torch
import csv
import numpy as np
import pandas as pd
from nltk import sent_tokenize
limit = 50
# Data Class
class Dictionary( object ):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append( word )
            self.word2idx[ word ] = len(self.idx2word) - 1
        return self.word2idx[ word ]

    def __len__(self):
        return len( self.idx2word )


class Corpus( object ):
    def __init__(self, path):
        self.dictionary = Dictionary()

        # Load all train, valid and test data
        self.train, self.train_len, self.train_label = self.tokenize( os.path.join( path, 'train.csv' ) )
        self.valid, self.valid_len, self.valid_label = self.tokenize( os.path.join( path, 'valid.csv' ) )
        self.test, self.test_len, self.test_label = self.tokenize( os.path.join( path, 'test.csv' ) )

    def tokenize( self, path ):
        """Tokenizes a csv file."""
        assert os.path.exists( path )
        # Add words to the dictionary
        max_len = 0
        with open( path, 'r' ) as f:
            Reader = csv.reader( f, delimiter=',', quoting=csv.QUOTE_MINIMAL )
            count = 0
            for record in Reader:
                count += 1
                # review = record[ 0 ].decode( 'utf-8' )
                review = record[ 0 ]
                # words = [ word for sent in sent_tokenize( review ) for word in word_tokenize( sent ) ]
                words = review.split()[0:limit]
                if len( words ) > max_len:
                    max_len = len( words )
                for word in words:
                    self.dictionary.add_word( word )

        # Tokenize file content
        with open( path, 'r' ) as f:
            Reader = csv.reader( f, delimiter=',', quoting=csv.QUOTE_MINIMAL )
            # data = torch.LongTensor( count, max_len ).fill_( 0 )
            # data = np.zeros((count, max_len), dtype=np.int)
            data = []
            lengths = []
            labels = []
            # labels = np.zeros(count, dtype=np.int)

            '''for idx, record in enumerate( Reader ):
                labels[ idx ] = int( record[ 1 ] )
                review = record[ 0 ]
                # words = [ word for sent in sent_tokenize( review ) for word in word_tokenize( sent ) ]
                words = review.split()
                lengths.append( len( words ) )

                for i, word in enumerate( words ):
                    data[ idx, i ] = self.dictionary.word2idx[word]'''
            n_count = 0
            for _, record in enumerate(Reader):
                if n_count >= 10:
                    break
                text = record[0]
                words = text.split()[0:limit]
                data.append([self.dictionary.word2idx[w] for w in words])
                labels.append(int(record[1]))
                lengths.append(len(words))
                '''sents = sent_tokenize(text=text)
                for sent in sents:
                    words = sent.split()
                    if len(words) < 5:
                        continue
                    lengths.append(len(words))
                    labels.append(int(record[1]))
                    t_sent = []
                    for i, word in enumerate(sent.split()):
                        t_sent.append(self.dictionary.word2idx[word])
                        try:
                            data[n_count, i] = self.dictionary.word2idx[word]
                        except Exception:
                            print(word)
                            len(self.dictionary.word2idx)
                    data.append(t_sent)
                    n_count += 1'''
        data = pd.DataFrame(data).fillna(0).as_matrix()
        lengths = np.asarray(lengths)
        labels = np.asarray(labels)
        assert data.shape[0] == lengths.shape[0] == labels.shape[0]
        return data, lengths, labels
