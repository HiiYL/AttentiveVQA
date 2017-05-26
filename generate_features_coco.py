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

def train(save_path, args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Scale((args.crop_size, args.crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    coco = CocoImgDataset(
        image_dir="/media/citi/afb54f66-7906-4a61-8711-eb01902c9faf/Images/mscoco/test2015",
         annotation_dir="data/image_info_test-dev2015.json",
         transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=args.num_workers)
    # Build the models
    encoder = EncoderCNN(args.embed_size,models.inception_v3(pretrained=True), requires_grad=False)

    if torch.cuda.is_available():
        encoder = encoder.cuda()

    save_path = "data/features_testdev_598"

    # Train the Models
    total_step = len(data_loader)
    total_iterations = 0
    encoder.eval()
    for (images,img_id) in tqdm(data_loader):
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
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=598 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--dataset', type=str, default='aesthetics' ,
                        help='dataset to use')
    parser.add_argument('--comments_path', type=str,
                        default='data/labels.h5',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--tb_log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=10000,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=3 ,
                        help='number of layers in lstm')
    parser.add_argument('--pretrained', type=str)#, default='-2-20000.pkl')
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clip', type=float, default=1.0,help='gradient clipping')
    args = parser.parse_args()
    print(args)

    if not os.path.exists("logs"):
        os.mkdir("logs")

    if not os.path.exists(os.path.join("logs", args.dataset)):
        os.mkdir(os.path.join("logs", args.dataset))

    now = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
    save_path = os.path.join(os.path.join("logs", args.dataset), now)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    configure(save_path)
    train(save_path, args)