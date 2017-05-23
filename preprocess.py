import json
import nltk
from collections import Counter
import cPickle as pickle
from build_vocab import Vocabulary
import spacy

from tqdm import tqdm

topk                   = 2000
vqa_val_save_path      = "data/vqa_val.json"
vqa_train_save_path    = "data/vqa_train.json"
vqa_combined_save_path = "data/vqa_combined.json"
vqa_test_save_path     = "data/vqa_test.json"

question_vocab_path = 'data/question_vocab.pkl'
ans_vocab_save_path = 'data/ans_vocab.pkl'

split = 2

print("Loading Annotations ...")
coco_caption = json.load(open('data/captions_train2014.json', 'r'))
coco_caption_val = json.load(open('data/captions_val2014.json', 'r'))
coco_images_test = json.load(open('data/image_info_test2015.json', 'r'))

train_anno = json.load(open("data/Annotations/v2_mscoco_train2014_annotations.json", "r"))
train_ques = json.load(open("data/Questions/v2_OpenEnded_mscoco_train2014_questions.json", "r"))

val_anno   = json.load(open("data/Annotations/v2_mscoco_val2014_annotations.json", "r"))
val_ques   = json.load(open("data/Questions/v2_OpenEnded_mscoco_val2014_questions.json", "r"))

test_ques = json.load(open("data/Questions/v2_OpenEnded_mscoco_test-dev2015_questions.json", "r"))


def prepare_data(coco_data, ques, anno=None):
    vqa = {}
    if anno is None:
        for question in ques["questions"]:
            question["id"] = question["question_id"]
        vqa["annotations"] = ques["questions"]
    else:
        annotations = []
        for i in tqdm(range(len(anno['annotations']))):
            ans = anno['annotations'][i]['multiple_choice_answer']
            question_id = anno['annotations'][i]['question_id']
            image_id = anno['annotations'][i]['image_id']

            question = ques['questions'][i]['question']
            mc_ans   = anno['annotations'][i]['answers']
            annotations.append({'id': question_id, 'image_id': image_id, 'question': question,'ans': ans}) #, 'MC_ans': mc_ans})
        vqa["annotations"] = annotations
    
    vqa["images"] =  coco_data["images"]
    return vqa


def merge(vqa_1, vqa_2):
    vqa_merged = {}
    vqa_merged["annotations"] = vqa_1["annotations"] + vqa_2["annotations"]
    vqa_merged["images"] = vqa_1["images"] + vqa_2["images"]

    return vqa_merged

def prepare_answers_vocab(vqa):
    s = 'ans'
    to_process = vqa["annotations"]
    
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

    return ans_vocab, common_words

def prepare_question_vocab(vqa):
    s = 'question'
    to_process = vqa["annotations"] 
    counter = Counter()
    question_vocab = Vocabulary()
    print("Generating question vocabulary")

    for i, qa_sample in enumerate(tqdm(to_process)):
        caption = str(qa_sample[s])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        for token in tokens:
            question_vocab.add_word(token)

        qa_sample["processed_tokens"] = tokens

    question_vocab.add_word('<end>')
    question_vocab.add_word('<unk>')

    return question_vocab

def convert_word_to_idx(vqa_annotations, question_vocab):
    print("Converting word to indices ... ")
    for i, qa_sample in enumerate(tqdm(vqa_annotations)):
        if "processed_tokens" in qa_sample:
            tokens =  qa_sample["processed_tokens"]
        else:
            caption = str(qa_sample['question'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            qa_sample["processed_tokens"] = tokens

        question = []
        question.extend([question_vocab(token) for token in tokens])
        question.append(question_vocab('<end>'))
        qa_sample["final_question"] = question

    return vqa_annotations


def trim(to_process, common_words, s='ans'):
    initial_length = len(to_process)
    trimmed = []
    for i, qa_sample in enumerate(tqdm(to_process)):
        if qa_sample[s] in common_words:
            trimmed.append(qa_sample)

    difference = initial_length - len(trimmed)
    percentage_remaining = ( len(trimmed) / float(initial_length) ) * 100
    print("")
    print("question number reduced from {} to {} ( {:.2f} % )"
        .format(initial_length,  len(trimmed) , percentage_remaining))
    return trimmed


if __name__ == '__main__':
    if split == 1:
        print("--------------------------------")
        print("| Train -> Train | Test -> Val |")
        print("--------------------------------")
        print("Preparing Training Data ... ")
        vqa_train = prepare_data(coco_caption, train_ques, train_anno)

        print("Preparing Validation Data ... ")
        vqa_val   = prepare_data(coco_caption_val, val_ques, val_anno)

        print("Preparing Answers Vocab ... ")
        ans_vocab, common_words = prepare_answers_vocab(vqa_train)

        print("Counting training samples to remove ... ")
        vqa_train["annotations"] = trim(vqa_train["annotations"], common_words)

        print("Preparing Questions Vocab ... ")
        question_vocab = prepare_question_vocab(vqa_train)

        print("Saving VQA training data ...")
        json.dump(vqa_train, open(vqa_train_save_path, "w"))
        print("vqa_train saved to " + vqa_train_save_path)

        json.dump(vqa_val, open(vqa_val_save_path, "w"))
        print("vqa_val saved to " + vqa_val_save_path)

    elif split == 2:
        print("---------------------------------------")
        print("| Train -> Train + Val | Test -> Test |")
        print("---------------------------------------")
        print("Preparing Training Data ... ")
        vqa_train = prepare_data(coco_caption    , train_ques, train_anno)
        vqa_val   = prepare_data(coco_caption_val, val_ques  , val_anno)

        vqa_combined = merge(vqa_train, vqa_val)

        print("Preparing Test Data")
        vqa_test  = prepare_data(coco_images_test, test_ques)
        
        print("Preparing Answers Vocab ... ")
        ans_vocab, common_words = prepare_answers_vocab(vqa_combined)

        print("Counting training samples to remove ... ")
        vqa_combined["annotations"] = trim(vqa_combined["annotations"], common_words)

        print("Preparing Questions Vocab ... ")
        question_vocab = prepare_question_vocab(vqa_combined)

        vqa_combined["annotations"]   = convert_word_to_idx(vqa_combined["annotations"], question_vocab)
        vqa_test["annotations"]       = convert_word_to_idx(vqa_test["annotations"], question_vocab)

        print("Saving VQA training data ...")
        json.dump(vqa_combined, open(vqa_combined_save_path, "w"))
        print("vqa_combined saved to " + vqa_combined_save_path)

        print("Saving VQA test data ...")
        json.dump(vqa_test, open(vqa_test_save_path, "w"))
        print("vqa_test saved to " + vqa_test_save_path)



    print("Number of words in question vocab: {}".format(len(question_vocab)))
    with open(question_vocab_path, 'wb') as f:
        pickle.dump(question_vocab, f, 2)
    print("Question vocab saved to " + question_vocab_path)

    print("Number of words in answer vocab: {}".format(len(ans_vocab)))
    with open(ans_vocab_save_path, 'wb') as f:
        pickle.dump(ans_vocab, f, 2)
    print("Answer vocab saved to " + ans_vocab_save_path)