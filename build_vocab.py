import nltk
import pickle
import argparse
from collections import Counter
from pandas import HDFStore
import pandas as pd
import string

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(dataframe_path, threshold=5):
    """Build a simple vocabulary wrapper."""
    store = HDFStore(dataframe_path)
    ava_table = pd.concat([store['labels_train'], store['labels_test']])
    
    counter = Counter()
    for i, comment in enumerate(ava_table.comments):
        for comment in comment.split(" [END] "):
            tokens = nltk.tokenize.word_tokenize(comment.lower())
            counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ava_table)))


    print("Tokens found : %i"%(len(counter)))
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    print("Discarded %i tokens due to having frequency below threshold"%(len(counter) - len(words)))

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    chars = string.ascii_lowercase + string.digits + string.punctuation + string.whitespace
    # Adds the words to the vocabulary.
    for i, word in enumerate(chars):
        vocab.add_word(word)

    # for i in range(0,128):
    #     vocab.add_word(chr(i))
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_path', type=str, 
                        default='data/labels.h5', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)