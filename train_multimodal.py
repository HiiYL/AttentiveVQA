import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torchvision import transforms
from torchvision import models

cudnn.benchmark = True

import numpy as np
import os
from data_loader_coco import get_loader 
from build_vocab import Vocabulary
from models.encoder import EncoderCNN, EncoderRNN, EncoderFC, EncoderSkipThought
from models.classification_models import MultimodalAttentionRNN
import pickle
import datetime

import json

from tensorboard_logger import configure, log_value

from tools.PythonHelperTools.vqaTools.vqa import VQA
from tools.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
import json
import random
import os

from tqdm import tqdm, trange

def run(save_path, args):
    torch.manual_seed(args.seed)

    split = 1 # 1 -> train on train, test on val | 2 -> train on train+val, test on tesdev

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    train_transform = transforms.Compose([
        transforms.Scale((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    test_transform = transforms.Compose([
        transforms.Scale((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Load vocabulary wrapper.
    with open(args.question_vocab_path, 'rb') as f:
        question_vocab = pickle.load(f)

    with open(args.ans_vocab_path, 'rb') as f:
        ans_vocab = pickle.load(f)
    
    train_data_loader = get_loader("train", question_vocab, ans_vocab, "data/features_598/",
                             train_transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    validate = (split == 1)
    val_data_path = "data/features_598/" if split == 1 else "data/features_test_598"

    val_data_loader = get_loader("test", question_vocab, ans_vocab, val_data_path,
                             train_transform, args.val_batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Build the models
    netR = EncoderSkipThought(question_vocab)
    netM = MultimodalAttentionRNN(len(ans_vocab))

    if args.netR:
        print("[!]loading pretrained netR....")
        netR.load_state_dict(torch.load(args.netR))
        print("Done!")

    if args.netM:
        print("[!]loading pretrained decoder....")
        netM.load_state_dict(torch.load(args.netM))
        print("Done!")

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        netR.cuda()
        netM.cuda()
        criterion = criterion.cuda()

    params    = list(netR.parameters()) + list(netM.parameters())
    optimizer = torch.optim.RMSprop(params, lr=args.learning_rate)

    data_loader = iter(train_data_loader)

    total_iterations = 2500000
    t = trange(0, total_iterations)


    for iteration in t:
        try:
            (images, captions, lengths, ann_id, ans, relative_weights) = next(data_loader)
        except StopIteration:
            data_loader = iter(train_data_loader)
            (images, captions, lengths, ann_id, ans, relative_weights) = next(data_loader)

        # Set mini-batch dataset
        images        = Variable(images)
        captions      = Variable(captions)
        ans           = Variable(ans)
        relative_weights = list(relative_weights)

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            ans = ans.cuda()

        netR.zero_grad()
        netM.zero_grad()
        visual_features                  = images
        text_features, text_all_output   = netR(captions, lengths)
        out                              = netM(visual_features, text_features, text_all_output, lengths)

        loss = criterion(out, ans)

        # turn on for instant performance boooosstt
        #loss = 0
        #for i, relative_weight in enumerate(relative_weights):
        #    for (target, weight) in relative_weight:
        #        target = Variable(torch.cuda.LongTensor([target]))
        #        loss += weight * criterion(out[None,i], target)
        #loss /= len(relative_weights)

        loss.backward()
        torch.nn.utils.clip_grad_norm(params, args.clip)
        optimizer.step()


        if iteration == args.kick or iteration == args.kick * 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2 * param_group['lr']

        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.99997592083 * param_group['lr']

        # Print log info
        if iteration % args.log_step == 0:
            print('Step [%d/%d] Loss: %5.4f, Perplexity: %5.4f'
                  %(iteration, total_iterations,  loss.data[0], np.exp(loss.data[0])))


        if iteration % args.tb_log_step == 0:
            log_value('Loss', loss.data[0], iteration)
            log_value('Perplexity', np.exp(loss.data[0]), iteration)

        if (iteration + 1) % args.save_step == 0:
            torch.save(netR.state_dict(), os.path.join(save_path,'netR.pkl'))
            torch.save(netM.state_dict(), os.path.join(save_path,'netM.pkl'))
            export(netR, netM, val_data_loader,
             criterion, question_vocab,ans_vocab, iteration, save_path, validate=validate)

def export(netR, netM, data_loader, criterion,
    question_vocab,ans_vocab, iteration, save_path, validate=False):

    for net in [netR, netM]:
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad = False

    responses = []
    print("Exporting...")
    for i, (images, captions, lengths, ann_id) in enumerate(tqdm(data_loader)):
        # Set mini-batch dataset
        images = Variable(images, volatile=True)
        captions = Variable(captions, volatile=True)

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        visual_features = images
        text_features, text_all_output   = netR(captions, lengths)

        outputs = netM(visual_features, text_features, text_all_output, lengths)

        outputs = torch.max(outputs,1)[1]
        outputs = outputs.cpu().data.numpy().squeeze().tolist()

        for index in range(images.size(0)):
            answer = ans_vocab.idx2word[outputs[index]]
            responses.append({"answer":answer, "question_id": ann_id[index]})

    json_save_dir = os.path.join(save_path, "{}_OpenEnded_mscoco_val2014_fake_results.json".format(iteration))
    json.dump(responses, open(json_save_dir, "w"))

    print("")
    print("")

    if validate:
        dataDir     = 'data'
        taskType    ='OpenEnded'
        dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
        dataSubType ='val2014'
        annFile     ='%s/Annotations/v2_%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
        quesFile    ='%s/Questions/v2_%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
        imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
        resFile     = json_save_dir

        vqa     = VQA(annFile, quesFile)
        vqaRes  = vqa.loadRes(resFile, quesFile)
        vqaEval = VQAEval(vqa, vqaRes, n=2)
        vqaEval.evaluate()

        print "\n"
        print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
        log_value('Val_Acc', vqaEval.accuracy['overall'], iteration)

    for net in [netR, netM]:
        net.train()
        for parameter in net.parameters():
            parameter.requires_grad = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=299 ,
                        help='image size to use')
    parser.add_argument('--question_vocab_path', type=str, default='data/question_vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--ans_vocab_path', type=str, default='data/ans_vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--dataset', type=str, default='coco' ,
                        help='dataset to use')
    parser.add_argument('--comments_path', type=str,
                        default='data/labels.h5',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=100,
                        help='step size for prining log info')
    parser.add_argument('--tb_log_step', type=int , default=100,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=10000,
                        help='step size for saving trained models')
    parser.add_argument('--val_step', type=int , default=10000,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of gru hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in gru')
    parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
    parser.add_argument('--netM', type=str)
    parser.add_argument('--netR', type=str)

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--val_batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--kick', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=123)
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

    print("Logging to path: " + save_path)
    configure(save_path)
    run(save_path, args)