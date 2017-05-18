import json
import nltk
from collections import Counter
import pickle
from build_vocab import Vocabulary

coco_caption = json.load(open('data/captions_train2014.json', 'r'))
coco_caption_val = json.load(open('data/captions_val2014.json', 'r'))
coco_images_test = json.load(open('data/image_info_test-dev2015.json', 'r'))

train_anno = json.load(open("data/Annotations/v2_mscoco_train2014_annotations.json", "r"))
train_ques = json.load(open("data/Questions/v2_OpenEnded_mscoco_train2014_questions.json", "r"))

val_anno   = json.load(open("data/Annotations/v2_mscoco_val2014_annotations.json", "r"))
val_ques = json.load(open("data/Questions/v2_OpenEnded_mscoco_val2014_questions.json", "r"))

train = []
for i in range(len(train_anno['annotations'])):
    ans = train_anno['annotations'][i]['multiple_choice_answer']
    question_id = train_anno['annotations'][i]['question_id']
    image_id = train_anno['annotations'][i]['image_id']

    question = train_ques['questions'][i]['question']
    mc_ans   = train_anno['annotations'][i]['answers']

    train.append({'id': question_id, 'image_id': image_id, 'question': question,'ans': ans})#, 'MC_ans': mc_ans, })
vqa = {}
vqa["annotations"] = train
vqa["images"] = coco_caption["images"]
json.dump(vqa, open("data/vqa_train.json", "w"))

val = []
for i in range(len(val_anno['annotations'])):
    ans = val_anno['annotations'][i]['multiple_choice_answer']
    question_id = val_anno['annotations'][i]['question_id']
    image_id = val_anno['annotations'][i]['image_id']

    question = val_ques['questions'][i]['question']
    mc_ans   = val_anno['annotations'][i]['answers']

    val.append({'id': question_id, 'image_id': image_id, 'question': question,'ans': ans})#, 'MC_ans': mc_ans, })

vqa_val = {}
vqa_val["annotations"] = val
vqa_val["images"] = coco_caption_val["images"]
json.dump(vqa_val, open("data/vqa_val.json", "w"))

test_ques = json.load(open("data/Questions/v2_OpenEnded_mscoco_test-dev2015_questions.json", "r"))
vqa_test = {}
for question in test_ques["questions"]:
    question["id"] = question["question_id"]

vqa_test["annotations"] = test_ques["questions"]
vqa_test["images"] = coco_images_test["images"]
json.dump(vqa_test, open("data/vqa_test.json", "w"))


s = 'question'
counter = Counter()
for i, qa_sample in enumerate(train):
    caption = str(qa_sample[s])
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    counter.update(tokens)

    if i % 1000 == 0:
        print("[%d/%d] Tokenized the captions." %(i, len(train)))
threshold=5
common_words = [word for word, cnt in counter.items() if cnt >= threshold]
# Creates a vocab wrapper and add some special tokens.
vocab = Vocabulary()
vocab.add_word('<pad>')
vocab.add_word('<start>')
vocab.add_word('<end>')
vocab.add_word('<unk>')
# Adds the words to the vocabulary.
for i, word in enumerate(common_words):
    vocab.add_word(word)
with open('data/{}_vocab.pkl'.format(s), 'wb') as f:
    pickle.dump(vocab, f, 2)

s = 'ans'
counter = Counter()
for i, qa_sample in enumerate(train):
    caption = str(qa_sample[s])
    counter.update([caption])

    if i % 1000 == 0:
        print("[%d/%d] Tokenized the captions." %(i, len(train)))

#common_words = [ word for word,cnt in counter.items() ]
common_words = [ word for word,cnt in counter.most_common(3000) ]
#threshold=1
#common_words = [word for word, cnt in counter.items() if cnt >= threshold]
# Creates a vocab wrapper and add some special tokens.
vocab = Vocabulary()
vocab.add_word('<unk>')
# Adds the words to the vocabulary.
for i, word in enumerate(common_words):
    vocab.add_word(word)
with open('data/{}_vocab.pkl'.format(s), 'wb') as f:
    pickle.dump(vocab, f, 2)