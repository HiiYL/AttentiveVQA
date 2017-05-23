import json
import nltk
from collections import Counter
import pickle
from build_vocab import Vocabulary

coco_caption = json.load(open('data/captions_train2014.json', 'r'))
coco_caption_val = json.load(open('data/captions_val2014.json', 'r'))
coco_images_test = json.load(open('data/image_info_test2015.json', 'r'))

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
#json.dump(vqa, open("data/vqa_train.json", "w"))

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
# json.dump(vqa_val, open("data/vqa_val.json", "w"))

# vqa_complete = {}
# vqa_complete["annotations"] = vqa["annotations"] + vqa_val["annotations"]
# vqa_complete["images"] = vqa["images"] + vqa_val["images"]
# json.dump(vqa_complete, open("data/vqa_train+val_complete.json", "w"))

test_ques = json.load(open("data/Questions/v2_OpenEnded_mscoco_test-dev2015_questions.json", "r"))
vqa_test = {}
for question in test_ques["questions"]:
    question["id"] = question["question_id"]

vqa_test["annotations"] = test_ques["questions"]
vqa_test["images"] = coco_images_test["images"]
json.dump(vqa_test, open("data/vqa_test.json", "w"))


image_ids = []
for question in test_ques["questions"]:
    image_ids.append(question['image_id'])

coco_image_ids = []
for image in coco_images_test["images"]:
    coco_image_ids.append(image['id'])

coco_image_ids = []
for image in coco_images_test["images"]:
    coco_image_ids.append(image['id'])

test_images_uniques_count = len(set(image_ids))
test_images_overlap_count = len(set(image_ids).intersection(set(coco_image_ids)))
assert test_images_uniques_count == test_images_overlap_count \
 , " {} Missing images in test image set".format(abs(test_images_overlap_count - test_images_uniques_count))


print("Preparing Answers vocab ... ")
s = 'ans'
to_process = train + val
counter = Counter()
for i, qa_sample in enumerate(to_process):
    caption = str(qa_sample[s])
    counter.update([caption])
    if i % 1000 == 0:
        print("[%d/%d] Tokenized the captions." %(i, len(to_process)))

common_words = [ word for word,cnt in counter.most_common(1000) ]
vocab = Vocabulary()
vocab.add_word('<unk>')
# Adds the words to the vocabulary.
for i, word in enumerate(common_words):
    vocab.add_word(word)
with open('data/{}_vocab.pkl'.format(s), 'wb') as f:
    pickle.dump(vocab, f, 2)


print("Removing UNK entries from dataset ...")
def removeUNK(to_process, common_words):
    initial_length = len(to_process)
    train_without_unk = []
    for i, qa_sample in enumerate(to_process):
        if qa_sample['ans'] in common_words:
            train_without_unk.append(qa_sample)

        if i % 1000 == 0:
            print("[%d/%d] Filtering unks." %(i, len(to_process)))
    print("{} out of {} answers containing <unk> discarded".format(initial_length - len(train_without_unk), initial_length))
    return train_without_unk

vqa_combined = {}
vqa_combined["images"] = coco_caption["images"] + coco_caption_val["images"]
combined_annotations_without_unk = removeUNK(train + val, common_words)
vqa_combined["annotations"] = combined_annotations_without_unk
json.dump(vqa_combined, open("data/vqa_train+val.json", "w"))


print("Preparing Questions vocab ... ")
s = 'question'
to_process = combined_annotations_without_unk #train_without_unk #+ val
counter = Counter()
# max_length = 0
for i, qa_sample in enumerate(to_process):
    caption = str(qa_sample[s])
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    counter.update(tokens)
    # if len(tokens) > max_length:
    #     max_length = len(tokens)
    #     print(max_length)

    if i % 1000 == 0:
        print("[%d/%d] Tokenized the captions." %(i, len(to_process)))
threshold=0
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
print("Question vocab contains {} words".format(len(vocab)))
with open('data/{}_vocab.pkl'.format(s), 'wb') as f:
    pickle.dump(vocab, f, 2)


# print("Preparing Validation Answers vocab ... ")
# s = 'ans'
# to_process = val

# vocab = Vocabulary()
# vocab.add_word('<unk>')
# counter = Counter()
# for i, qa_sample in enumerate(to_process):
#     caption = str(qa_sample[s])
#     counter.update([caption])
    
#     if i % 1000 == 0:
#         print("[%d/%d] Tokenized the captions." %(i, len(to_process)))

# for i, word in enumerate(common_words):
#     vocab.add_word(word)

# print("{} unique entries in val vocab".format(len(vocab)))
# with open('data/val_{}_vocab.pkl'.format(s), 'wb') as f:
#     pickle.dump(vocab, f, 2)


chars = string.ascii_lowercase + string.digits + string.punctuation + ' '
vocab = Vocabulary()
vocab.add_word('<pad>')
vocab.add_word('<start>')
vocab.add_word('<end>')
vocab.add_word('<unk>')
# Adds the words to the vocabulary.
for i, word in enumerate(chars):
    vocab.add_word(word)

with open('data/ans_vocab_char.pkl'.format(s), 'wb') as f:
    pickle.dump(vocab, f, 2)