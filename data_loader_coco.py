import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import cPickle as pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO

import random
import json

import h5py


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, mode, question_vocab, ans_vocab, finetune=False,transform=None, classification=True):
        if mode == "test":
            self.root = "data/Images/mscoco/test2015"
            self.pickle_feature_path = "data/features_test/"
        else:
            self.root = "data/Images/mscoco/merged2014"
            self.pickle_feature_path = "data/features/"
        
        self.coco = COCO("data/vqa_{}.json".format(mode))
        self.ids = list(self.coco.anns.keys())

        self.question_vocab = question_vocab
        self.ans_vocab = ans_vocab

        self.transform = transform
        self.finetune  = finetune
        self.mode      = mode
        self.classification = classification

        self.question_type_vocab = pickle.load(open("data/question_type_vocab.pkl", "r"))

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco           = self.coco
        question_vocab = self.question_vocab
        ans_vocab      = self.ans_vocab
        ann_id         = self.ids[index]

        question = coco.anns[ann_id]['final_question']
        img_id   = coco.anns[ann_id]['image_id']

        if self.finetune:
            path = coco.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(self.root, path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        else:
            filename = str(img_id) + ".npz"
            path = os.path.join(self.pickle_feature_path, filename)
            image = np.load(path)['arr_0']
            image = torch.from_numpy(image)

        question = torch.Tensor(question)

        if self.mode in ["test", "val"]:
            return image, question, ann_id

        ans      = coco.anns[ann_id]['ans']
        qtype    = coco.anns[ann_id]['question_type']

        if self.classification:
            ans = torch.LongTensor([ans])
        else:
            tokens = nltk.tokenize.word_tokenize(str(ans).lower())
            ans = []
            ans.append(ans_vocab('<start>'))
            ans.extend([ans_vocab(token) for token in tokens])
            ans.append(ans_vocab('<end>'))

        question_type = torch.LongTensor([qtype])


        return image, question, ann_id, ans, question_type

    def __len__(self):
        return len(self.ids)

def collate_fn_test(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ann_id = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ann_id

def collate_fn_vqa(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ann_id, ans, question_type = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images           = torch.stack(images, 0)
    ans              = torch.cat(ans, 0)
    question_type    = torch.cat(question_type, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]


    # ans_lengths = [len(cap) for cap in ans]
    # ans_targets = torch.zeros(len(ans), max(ans_lengths)).long()
    # for i, cap in enumerate(ans):
    #     end = ans_lengths[i]
    #     ans_targets[i, :end] = cap[:end]

    return images, targets, lengths, ann_id,ans, question_type #ans_targets, ans_lengths

def get_loader(mode, question_vocab,ans_vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    print(mode)
    # COCO caption dataset
    coco = CocoDataset(mode=mode,
                   question_vocab=question_vocab,
                   ans_vocab=ans_vocab,
                   transform=transform)
    
    if mode in ["test", "val"]:
        collate_fn_to_use = collate_fn_test
    else:
        collate_fn_to_use = collate_fn_vqa
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn_to_use)
    return data_loader