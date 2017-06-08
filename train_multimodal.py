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
from models.encoder import EncoderCNN, EncoderSkipThought
from models.classification_models import MultimodalAttentionRNN
from config import get_config
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

class Trainer():
    def __init__(self,args):
        self.args = args
        torch.manual_seed(args.seed)

        # Create model directory
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        self._prepare_tensorboard_dir(args)
        self._load_vocab(args)
        self._prepare_dataloader(args)

        # Build the models
        self.netR = EncoderSkipThought(self.question_vocab)
        self.netM = MultimodalAttentionRNN(self.ans_vocab)

        if args.netR:
            print("[!]loading pretrained netR....")
            self.netR.load_state_dict(torch.load(args.netR))
            print("Done!")

        if args.netM:
            print("[!]loading pretrained decoder....")
            self.netM.load_state_dict(torch.load(args.netM))
            print("Done!")

        self.criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self.netR.cuda()
            self.netM.cuda()
            self.criterion.cuda()

        self.params    = list(self.netR.parameters()) + list(self.netM.parameters())
        self.optimizer = torch.optim.RMSprop(self.params, lr=args.learning_rate)


    def _load_vocab(self, args):
        # Load vocabulary wrapper.
        with open(args.vocabs_path, 'rb') as f:
            vocabs = pickle.load(f)

        self.question_vocab = vocabs["question"]
        self.ans_vocab      = vocabs["answer"]
        self.ans_type_vocab = vocabs["ans_type"]

    def _prepare_dataloader(self, args):
        split = args.split
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
        
        self.train_data_loader = get_loader("train", self.question_vocab, self.ans_vocab, "data/features_resnet_448/",
                                 train_transform, args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)

        self.validate = (split == 1)
        val_data_path = "data/features_resnet_448/" if split == 1 else "data/features_resnet_448_test"

        self.val_data_loader = get_loader("test", self.question_vocab, self.ans_vocab, val_data_path,
                                 train_transform, args.val_batch_size,
                                 shuffle=False, num_workers=args.num_workers)

    def _prepare_tensorboard_dir(self,args):
        if not os.path.exists("logs"):
            os.mkdir("logs")

        if not os.path.exists(os.path.join("logs", args.dataset)):
            os.mkdir(os.path.join("logs", args.dataset))

        now = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
        self.save_path = os.path.join(os.path.join("logs", args.dataset), now)

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        print("Logging to path: " + self.save_path)
        configure(self.save_path)

    def run(self):
        args = self.args
        data_loader = iter(self.train_data_loader)

        total_iterations = 2500000
        t = trange(0, total_iterations)

        for iteration in t:
            try:
                data = next(data_loader)
            except StopIteration:
                data_loader = iter(self.train_data_loader)
                data = next(data_loader)

            (images, captions, lengths, ann_id, ans, ans_type, relative_weights) = data
            # Set mini-batch dataset
            images        = Variable(images)
            captions      = Variable(captions)
            ans           = Variable(ans)
            #ans_type      = list(ans_type)
            ans_type      = Variable(ans_type)
            relative_weights = list(relative_weights)

            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                ans = ans.cuda()
                ans_type = ans_type.cuda()

            self.netR.zero_grad()
            self.netM.zero_grad()
            visual_features                  = images
            text_features, text_all_output   = self.netR(captions, lengths)
            out                              = self.netM(visual_features, text_features, text_all_output, lengths)

            loss     = self.criterion(out, ans)
            #aux_loss = self.riterion(aux, ans_type)

            #turn on for instant performance boooosstt
            # loss = 0
            # for i, relative_weight in enumerate(relative_weights):
            #     for (target, weight) in relative_weight:
            #         target = Variable(torch.cuda.LongTensor([target]))
            #         loss += weight * self.criterion(out[None,i], target)
            # loss /= len(relative_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.params, args.clip)
            self.optimizer.step()

            if iteration == args.kick or iteration == args.kick * 2:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 2 * param_group['lr']

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.99997592083 * param_group['lr']

            # Print log info
            if iteration % args.log_step == 0:
                print('Step [%d/%d] Loss: %5.4f'
                      %(iteration, total_iterations,  loss.data[0]))


            if iteration % args.tb_log_step == 0:
                log_value('Loss', loss.data[0], iteration)
                log_value('Perplexity', np.exp(loss.data[0]), iteration)

            if (iteration + 1) % args.save_step == 0:
                torch.save(self.netR.state_dict(), os.path.join(self.save_path,'netR.pkl'))
                torch.save(self.netM.state_dict(), os.path.join(self.save_path,'netM.pkl'))
                self.export(iteration)

    def export(self, iteration):
        args = self.args
        for net in [self.netR, self.netM]:
            net.eval()
            for parameter in net.parameters():
                parameter.requires_grad = False

        responses = []
        print("Exporting...")
        for (images, captions, lengths, ann_id) in tqdm(self.val_data_loader):
            # Set mini-batch dataset
            images = Variable(images, volatile=True)
            captions = Variable(captions, volatile=True)

            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()

            visual_features = images
            text_features, text_all_output   = self.netR(captions, lengths)

            outputs = self.netM(visual_features, text_features, text_all_output, lengths)

            outputs = torch.max(outputs,1)[1]
            outputs = outputs.cpu().data.numpy().squeeze(1).tolist()

            for index in range(images.size(0)):
                current_answer_idx = outputs[index]
                answer = self.ans_vocab.idx2word[current_answer_idx]
                responses.append({"answer":answer, "question_id": ann_id[index]})

        json_save_dir = os.path.join(self.save_path, "{}_OpenEnded_mscoco_val2014_fake_results.json".format(iteration))
        json.dump(responses, open(json_save_dir, "w"))

        print("")
        print("")

        if self.validate:
            dataDir     = 'data'
            taskType    ='OpenEnded'
            dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
            dataSubType ='val2014'
            if args.split == 3:
                annFile     = "data/val_annotations_trimmed.json"
                quesFile    = "data/val_questions_trimmed.json"
            else:
                annFile     ='%s/Annotations/v2_%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
                quesFile    ='%s/Questions/v2_%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)

            imgDir      = '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
            resFile     = json_save_dir

            vqa     = VQA(annFile, quesFile)
            vqaRes  = vqa.loadRes(resFile, quesFile)
            vqaEval = VQAEval(vqa, vqaRes, n=2)
            vqaEval.evaluate()

            print "\n"
            print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
            log_value('Val_Acc', vqaEval.accuracy['overall'], iteration)

        for net in [self.netR, self.netM]:
            net.train()
            for parameter in net.parameters():
                parameter.requires_grad = True


if __name__ == '__main__':
    args = get_config()
    print(args)

    trainer = Trainer(args)
    trainer.run()