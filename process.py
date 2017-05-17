import json
import nltk
from collections import Counter
import pickle

coco_caption = json.load(open('data/captions_train2014.json', 'r'))
train_anno     = json.load(open("v2_mscoco_train2014_annotations.json", "r"))
train_ques = json.load(open("v2_OpenEnded_mscoco_train2014_questions.json", "r"))

len(train_ques['questions'])
len(train_anno['annotations'])


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


train = []
for i in range(len(train_anno['annotations'])):
    ans = train_anno['annotations'][i]['multiple_choice_answer']
    question_id = train_anno['annotations'][i]['question_id']
    image_id = train_anno['annotations'][i]['image_id']

    question = train_ques['questions'][i]['question']
    mc_ans = train_anno['annotations'][i]['answers']

    train.append({'id': question_id, 'image_id': image_id, 'question': question,'ans': ans})#, 'MC_ans': mc_ans, })

vqa = {}
vqa["annotations"] = train
vqa["images"] = coco_caption["images"]
json.dump(vqa, open("data/vqa_train.json", "w"))
for s in ['ans', 'question']:
    counter = Counter()
    for i, qa_sample in enumerate(train):
        caption = str(qa_sample[s])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(train)))
    threshold=5
    ans_words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    with open('{}_vocab.pkl'.format(s), 'wb') as f:
        pickle.dump(vocab, f, 2)