import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch.utils.data as data
from pycocotools.coco import COCO
from PIL import Image

cudnn.benchmark = True

import numpy as np
import os
from build_vocab import Vocabulary
from models.encoder import EncoderCNN
import cPickle as pickle
import datetime
import torchvision.datasets as dset
from torchvision import models

import time
from tqdm import tqdm

from tensorboard_logger import configure, log_value
import h5py

class CocoImgDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, image_dir, annotation_dir, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = image_dir
        self.coco = COCO(annotation_dir)
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco

        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, img_id

    def __len__(self):
        return len(self.ids)

def generate_features(args):
    model_resolutions = { "inception_v3": 299, "resnet": 224 }
    crop_size = model_resolutions[args.model] * args.scale
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Scale((crop_size, crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_coco = CocoImgDataset(
        image_dir="/media/citi/afb54f66-7906-4a61-8711-eb01902c9faf/Images/mscoco/merged2014",
         annotation_dir="data/vqa_train.json",
         transform=transform)

    test_data_loader = torch.utils.data.DataLoader(dataset=test_coco, 
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=args.num_workers)
    # Build the models
    if args.model == "inception":
        model = models.inception_v3(pretrained=True)
        encoder = EncoderCNN(model, requires_grad=False)
    elif args.model == "resnet":
        encoder =  nn.Sequential(*list(models.resnet152(pretrained=True).children())[:-2]) 


    

    if torch.cuda.is_available():
        encoder = encoder.cuda()

    save_path = "data/features_{}_{}".format(args.model, crop_size)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Train the Models
    encoder.eval()
    for (images,img_id) in tqdm(test_data_loader):
        # Set mini-batch dataset
        images = Variable(images, volatile=True)
        if torch.cuda.is_available():
            images = images.cuda()

        features = encoder(images)
        for j in range(images.size(0)):
            key = str(img_id.numpy()[j])
            path = os.path.join(save_path, key)
            value = features[j].data.cpu().float().numpy()
            np.savez_compressed(path, value)

    #f.close()


                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['inception','resnet'] , help='type of model to use')
    parser.add_argument('--scale', type=int, default=2 , help='input image resolution multiplier')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    print(args)

    generate_features(args)