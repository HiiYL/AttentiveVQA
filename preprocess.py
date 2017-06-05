import json
import nltk
from collections import Counter
import cPickle as pickle
from build_vocab import Vocabulary

from tqdm import tqdm
import spacy

vqa_train_save_path           = "data/vqa_train.json"
vqa_test_save_path            = "data/vqa_test.json"
vocabs_save_path              = 'data/vocabs.pkl'

topk        = 2000
split       = 3 # 1 - train->train, test->val | 2 - train->train+val | 3 - karpathy's split

print("Loading Annotations ...")
coco_caption     = json.load(open('data/captions_train2014.json', 'r'))
coco_caption_val = json.load(open('data/captions_val2014.json', 'r'))
coco_images_test = json.load(open('data/image_info_test2015.json', 'r'))

train_anno = json.load(open("data/Annotations/v2_mscoco_train2014_annotations.json", "r"))
val_anno   = json.load(open("data/Annotations/v2_mscoco_val2014_annotations.json", "r"))

train_ques = json.load(open("data/Questions/v2_OpenEnded_mscoco_train2014_questions.json", "r"))
val_ques   = json.load(open("data/Questions/v2_OpenEnded_mscoco_val2014_questions.json", "r"))
test_ques  = json.load(open("data/Questions/v2_OpenEnded_mscoco_test-dev2015_questions.json", "r"))


def prepare_data(coco_data, questions, annotations=None):
    vqa = {}
    if annotations is None:
        for question in questions:
            question["id"] = question["question_id"]
        vqa["annotations"] = questions
    else:
        for i, annotation in enumerate(tqdm(annotations)):
            annotation['question']  = questions[i]['question']
            annotation['id']        = annotation['question_id']
            annotation['answers']   = [ item['answer'] for item in annotation['answers'] ]
        vqa["annotations"] = annotations
    
    vqa["images"] =  coco_data
    return vqa

def merge(dict_1, dict_2):
    merged = {}
    for key in dict_1.keys():
        if isinstance(dict_1[key], list):
            merged[key] = dict_1[key] + dict_2[key]
    return merged

def prepare_answers_vocab(annotations, topk):
    s = 'multiple_choice_answer'
    
    counter = Counter()
    print("counting occurrences ... ")
    for annotation in tqdm(annotations):
        caption = str(annotation[s])
        counter.update([caption])

    print("Top 20 Most Common Words")
    for word, cnt in counter.most_common(20):
        print("{} - {}".format(word.ljust(10), cnt))
    common_words = [ word for word,cnt in counter.most_common(topk) ]
    #common_words = [ word for word,cnt in counter.iteritems() if cnt >= 100]

    ans_vocab = Vocabulary()
    for word in common_words:
        ans_vocab.add_word(word)

    return ans_vocab

def ans_type_to_idx(annotations):
    ans_type = []
    for anno in annotations:
        ans_type.append(anno["answer_type"])

    uniques = list(set(ans_type))
    ans_type_vocab = Vocabulary()

    for i, word in enumerate(uniques):
        ans_type_vocab.add_word(word)

    for anno in annotations:
        anno["answer_type"] = ans_type_vocab(anno["answer_type"])

    pickle.dump(ans_type_vocab, open("data/ans_type_vocab.pkl", "wb"))

    return ans_type_vocab

def prepare_question_type_vocab(annotations, topk=65):
    counter = Counter()
    print("counting question types ... ")

    for annotation in tqdm(annotations):
        caption = str(annotation['question_type'])
        counter.update([caption])

    print("Top 20 Most Common Questions")
    for word, cnt in counter.most_common(20):
        print("{} - {}".format(word.ljust(10), cnt))

    common_words = [ word for word,cnt in counter.most_common(topk) ]

    question_type_vocab = Vocabulary()
    for word in common_words:
        question_type_vocab.add_word(word)

    return question_type_vocab

def convert_field_to_index(annotations, vocab, field, out_field=None):
    for annotation in tqdm(annotations):
        current_input = annotation[field]
        out_field = out_field or field
        if isinstance(annotation[field], list):
            annotation[out_field] = [ vocab(item) for item in annotation[field] if item in vocab.word2idx ]
        else:
            annotation[out_field] = vocab(annotation[field])

    return annotations

def tokenize(annotations, field, out_field=None, tokenizer="spacy"):
    nlp = spacy.load('en')
    for annotation in tqdm(annotations):
        current_input = annotation[field]
        out_field = out_field or field
        if tokenizer == "spacy":
            tokens = [w.text for w in nlp(current_input)]
        else:
            caption = str(current_input)
            tokens = nltk.tokenize.word_tokenize(caption)

        annotation[out_field] = tokens

    return annotations

def prepare_question_vocab(annotations, field="question"):
    counter = Counter()
    question_vocab = Vocabulary()
    print("Generating question vocabulary")

    for annotation in tqdm(annotations):
        tokens = annotation[field]
        for token in tokens:
            question_vocab.add_word(token)

    question_vocab.add_word('<end>')
    question_vocab.add_word('<unk>')

    return question_vocab


def trim(annotations, ans_vocab, s='ans'):
    initial_length = len(annotations)
    trimmed = []
    words = set(ans_vocab.word2idx.keys())
    for annotation in tqdm(annotations):
        if annotation[s] in words:
            trimmed.append(annotation)

    difference = initial_length - len(trimmed)
    percentage_remaining = ( len(trimmed) / float(initial_length) ) * 100
    print("")
    print("question number reduced from {} to {} ( {:.2f} % )"
        .format(initial_length,  len(trimmed) , percentage_remaining))
    return trimmed


def relative_frequency(lst, element):
    return lst.count(element) / float(len(lst))

def calculate_confidence(annotations):
    for annotation in tqdm(annotations):
        uniques = set(annotation["answers"])
        relative_weights = []
        for unique in uniques:
            relative_weights.append((unique, relative_frequency(annotation["answers"], unique)))
        annotation["relative_weights"] = relative_weights

    return annotations

def trim_by_type(questions, annotations, keep="number"):
    keep_only = set()

    valid_anno = []
    for anno in annotations:
        if anno["answer_type"] == "number":
            valid_anno.append(anno)
            keep_only.add(anno["question_id"])

    valid_ques = []
    for question in questions:
        if question["question_id"] in keep_only:
            valid_ques.append(question)

    annotations = valid_anno
    questions   = valid_ques

    return questions, annotations

def karpathy_split(coco_caption,questions, annotations ):
    split = 5000
    
    test  = coco_caption["images"][:split]
    train = coco_caption["images"][split:]

    test_img_ids  = set( i["id"] for i in test )
    train_img_ids = set( i["id"] for i in train )

    train_inputs = {}
    test_inputs  = {}

    test_inputs["images"]      = test
    test_inputs["annotations"] = []
    test_inputs["questions"]   = []

    train_inputs["images"]      = train
    train_inputs["annotations"] = []
    train_inputs["questions"]   = []

    for annotation in annotations["annotations"]:
        if annotation["image_id"] in test_img_ids:
            test_inputs["annotations"].append(annotation)
        else:
            train_inputs["annotations"].append(annotation)

    for question in questions["questions"]:
        if question["image_id"] in test_img_ids:
            test_inputs["questions"].append(question)
        else:
            train_inputs["questions"].append(question)

    return train_inputs, test_inputs


if __name__ == '__main__':
    if split == 1:
        print("--------------------------------")
        print("| Train -> Train | Test -> Val |")
        print("--------------------------------")
        print("Preparing Training Data ... ")
        vqa_train = prepare_data(coco_caption, train_ques["questions"], train_anno["annotations"])

        print("Preparing Validation Data ... ")
        vqa_test   = prepare_data(coco_caption_val, val_ques["questions"])
    elif split == 2:
        print("---------------------------------------")
        print("| Train -> Train + Val | Test -> Test |")
        print("---------------------------------------")
        print("Preparing Training Data ... ")
        vqa_train = prepare_data(coco_caption    , train_ques["questions"], train_anno["annotations"])
        vqa_val   = prepare_data(coco_caption_val, val_ques["questions"]  , val_anno["annotations"])

        vqa_train = merge(vqa_train, vqa_val)

        print("Preparing Test Data")
        vqa_test  = prepare_data(coco_images_test, test_ques)
    elif split == 3:
        print("--------------------------------------")
        print("| Train -> K_Split | Test -> K_Split |")
        print("--------------------------------------")
        coco_captions = merge(coco_caption, coco_caption_val)
        questions     = merge(train_ques, val_ques)
        annotations   = merge(train_anno, val_anno)

        train_inputs, test_inputs = karpathy_split(coco_captions, questions, annotations)

        val_ques["questions"] = test_inputs["questions"]
        json.dump(val_ques, open("data/val_questions_trimmed.json", "wb"))

        val_anno["annotations"] = test_inputs["annotations"]
        json.dump(val_anno, open("data/val_annotations_trimmed.json", "wb"))

        print("Preparing Training Data ... ")
        vqa_train = prepare_data(train_inputs["images"], train_inputs["questions"], train_inputs["annotations"])

        print("Preparing Validation Data ... ")
        vqa_test = prepare_data(test_inputs["images"], test_inputs["questions"], test_inputs["annotations"])
        

    print("Adding answer type indices")
    ans_type_vocab = ans_type_to_idx(vqa_train["annotations"])

    print("Preparing Answers Vocab ... ")
    ans_vocab = prepare_answers_vocab(vqa_train["annotations"], topk)
    print("Counting training samples to remove ... ")
    vqa_train["annotations"] = trim(vqa_train["annotations"], ans_vocab, s='multiple_choice_answer')
    vqa_train["annotations"] = convert_field_to_index(vqa_train["annotations"], ans_vocab, "multiple_choice_answer")
    vqa_train["annotations"] = convert_field_to_index(vqa_train["annotations"], ans_vocab, "answers")

    print("Tokenizing....")
    vqa_train["annotations"] = tokenize(vqa_train["annotations"], "question")
    vqa_test["annotations"]  = tokenize(vqa_test["annotations"], "question")

    print("Preparing Questions Vocab ... ")
    question_vocab = prepare_question_vocab(vqa_train["annotations"])
    print("Preparing Question Type Vocab ... ")
    question_type_vocab      = prepare_question_type_vocab(vqa_train["annotations"])


    print("Converting fields to indices ... ")
    print("Processing question indices ...")
    vqa_train["annotations"] = convert_field_to_index(vqa_train["annotations"], question_vocab, "question")
    vqa_test["annotations"]  = convert_field_to_index(vqa_test["annotations"] , question_vocab, "question")

    print("Processing question type indices ...")
    vqa_train["annotations"] = convert_field_to_index(vqa_train["annotations"], question_type_vocab, "question_type")

    print("Adding MC_ans indices and calculating answer confidence")
    vqa_train["annotations"] = calculate_confidence(vqa_train["annotations"])


    ####
    #
    # Save
    #
    ####
    print("Saving VQA training data ...")
    json.dump(vqa_train, open(vqa_train_save_path, "w"))
    print("vqa_train saved to " + vqa_train_save_path)

    print("Saving VQA test data ...")
    json.dump(vqa_test, open(vqa_test_save_path, "w"))
    print("vqa_test saved to " + vqa_test_save_path)


    vocabs = {}
    vocabs["ans_type"]      = ans_type_vocab
    vocabs["question"]      = question_vocab
    vocabs["answer"]        = ans_vocab
    vocabs["question_type"] = question_type_vocab

    print("Saving vocabs...")
    for (name,vocab) in vocabs.iteritems():
        print("Number of words in {} vocab: {}".format(name, len(vocab)))

    with open(vocabs_save_path, 'wb') as f:
        pickle.dump(vocabs, f, 2)