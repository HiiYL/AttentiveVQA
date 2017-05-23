import json
import nltk
from collections import Counter
import cPickle as pickle
from build_vocab import Vocabulary

from tqdm import tqdm

topk                = 1000
vqa_val_save_path   = "data/vqa_val.json"
vqa_train_save_path = "data/vqa_train_top{}.json".format(topk)

question_vocab_path = 'data/question_vocab.pkl'
ans_vocab_save_path = 'data/ans_vocab.pkl'


print("Loading Annotations ...")
coco_caption = json.load(open('data/captions_train2014.json', 'r'))
coco_caption_val = json.load(open('data/captions_val2014.json', 'r'))
coco_images_test = json.load(open('data/image_info_test2015.json', 'r'))

train_anno = json.load(open("data/Annotations/v2_mscoco_train2014_annotations.json", "r"))
train_ques = json.load(open("data/Questions/v2_OpenEnded_mscoco_train2014_questions.json", "r"))

val_anno   = json.load(open("data/Annotations/v2_mscoco_val2014_annotations.json", "r"))
val_ques = json.load(open("data/Questions/v2_OpenEnded_mscoco_val2014_questions.json", "r"))


def prepare_data(ques, anno):
    train = []
    for i in tqdm(range(len(anno['annotations']))):
        ans = anno['annotations'][i]['multiple_choice_answer']
        question_id = anno['annotations'][i]['question_id']
        image_id = anno['annotations'][i]['image_id']

        question = ques['questions'][i]['question']
        mc_ans   = anno['annotations'][i]['answers']

        train.append({'id': question_id, 'image_id': image_id, 'question': question,'ans': ans, 'MC_ans': mc_ans})
    vqa = {}
    vqa["annotations"] = train
    vqa["images"] = coco_caption["images"]
    return vqa

def merge(vqa_1, vqa_2):
    vqa_complete = {}
    vqa_complete["annotations"] = vqa["annotations"] + vqa_val["annotations"]
    vqa_complete["images"] = vqa["images"] + vqa_val["images"]

    return vqa_complete

print("Preparing Training Data ... ")
vqa_train = prepare_data(train_ques, train_anno)

print("Preparing Validation Data ... ")
vqa_val   = prepare_data(val_ques, val_anno)



#json.dump(merge(vqa_train, vqa_val), open("data/vqa_train+val_complete.json", "w"))


print("Preparing Answers Vocab ... ")
s = 'ans'
to_process = vqa_train["annotations"]# + vqa_val["annotations"]


counter = Counter()
print("counting occurrences ... ")
for i, qa_sample in enumerate(tqdm(to_process)):
    caption = str(qa_sample[s])
    counter.update([caption])

print("Top 20 Most Common Words")
for word, cnt in counter.most_common(20):
    print("{} - {}".format(word.ljust(10), cnt))
common_words = [ word for word,cnt in counter.most_common(topk) ]

ans_vocab = Vocabulary()
for i, word in enumerate(common_words):
    ans_vocab.add_word(word)

print("Counting training samples to remove ... ")
def trim(to_process, common_words):
    initial_length = len(to_process)
    train_without_unk = []
    for i, qa_sample in enumerate(tqdm(to_process)):
        if qa_sample[s] in common_words:
            train_without_unk.append(qa_sample)

    difference = initial_length - len(train_without_unk)
    percentage_remaining = ( len(train_without_unk) / float(initial_length) ) * 100
    print("")
    print("question number reduced from {} to {} ( {:.2f} % )"
        .format(initial_length,  len(train_without_unk) , percentage_remaining))
    return train_without_unk


vqa_train["annotations"] = trim(vqa_train["annotations"], common_words)



print("Preparing Questions Vocab ... ")
s = 'question'
to_process = vqa_train["annotations"] 
counter = Counter()
question_vocab = Vocabulary()
for i, qa_sample in enumerate(tqdm(to_process)):
    caption = str(qa_sample[s])
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    for token in tokens:
        question_vocab.add_word(token)

    qa_sample["processed_tokens"] = tokens

question_vocab.add_word('<end>')
question_vocab.add_word('<unk>')

print("Number of words in question vocab: {}".format(len(question_vocab)))


with open(question_vocab_path, 'wb') as f:
    pickle.dump(question_vocab, f, 2)
print("Question vocab saved to " + question_vocab_path)

with open(ans_vocab_save_path, 'wb') as f:
    pickle.dump(ans_vocab, f, 2)
print("Answer vocab saved to " + ans_vocab_save_path)

json.dump(vqa_train, open(vqa_train_save_path, "w"))
print("vqa_train saved to " + vqa_train_save_path)

json.dump(vqa_val, open(vqa_val_save_path, "w"))
print("vqa_train saved to " + vqa_val_save_path)

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

def prepare_char_vocab():
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



def prepare_test():
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

    test_images_uniques_count = len(set(image_ids))
    test_images_overlap_count = len(set(image_ids).intersection(set(coco_image_ids)))
    assert test_images_uniques_count == test_images_overlap_count \
     , " {} Missing images in test image set".format(abs(test_images_overlap_count - test_images_uniques_count))