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
from data_loader_coco_generative import get_loader 
from build_vocab import Vocabulary
from models.encoder import EncoderRNN
from models.generative_models import G_Spatial_Adversarial
import pickle
import datetime

import json

from tensorboard_logger import configure, log_value



from tools.PythonHelperTools.vqaTools.vqa import VQA
from tools.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
import json
import random
import os

def run(save_path, args):
    torch.manual_seed(args.seed)

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

    # with open("data/val_ans_vocab.pkl", 'rb') asf:
    #     val_ans_vocab = pickle.load(f)
    
    train_data_loader = get_loader("train", question_vocab, ans_vocab,
                             train_transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # test_data_loader = get_loader("test", question_vocab, ans_vocab,
    #                          train_transform, args.val_batch_size,
    #                          shuffle=False, num_workers=args.num_workers)

    val_data_loader = get_loader("val", question_vocab, ans_vocab,
                             train_transform, args.val_batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Build the models
    # encoder = EncoderCNN(args.embed_size,models.inception_v3(pretrained=True), requires_grad=False)
    encoderRNN = EncoderRNN(args.embed_size, args.hidden_size, len(question_vocab), num_layers=2)
    netG = G_Spatial_Adversarial(args.embed_size, args.hidden_size, question_vocab, ans_vocab, args.num_layers)

    # if args.netG:
    #     print("[!]loading pretrained netG....")
    #     netG.load_state_dict(torch.load(args.netG))
    #     print("Done!")

    if args.encoder:
        print("[!]loading pretrained decoder....")
        encoder.load_state_dict(torch.load(args.encoder))
        print("Done!")

    criterion = nn.CrossEntropyLoss()
    
    state = (Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)),
     Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)))

    g2_state = (Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size * 4)),
     Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size * 4)))
    val_state = (Variable(torch.zeros(args.num_layers, args.val_batch_size, args.hidden_size)),
     Variable(torch.zeros(args.num_layers, args.val_batch_size, args.hidden_size)))

    #y_onehot = torch.FloatTensor(args.batch_size, 20,len(question_vocab))

    if torch.cuda.is_available():
        #encoder = encoder.cuda()
        encoderRNN.cuda()
        netG.cuda()
        #netD.cuda()
        state = [s.cuda() for s in state]
        g2_state = [s.cuda() for s in g2_state]
        val_state = [s.cuda() for s in val_state]
        criterion = criterion.cuda()
        #y_onehot = y_onehot.cuda()

    params = [
                {'params': netG.parameters()},
                #{'params': encoder.parameters(), 'lr': 0.1 * args.learning_rate}
                #{'params': encoder.fc.parameters()}
                
            ]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate,betas=(0.8, 0.999))

    # Train the Models
    total_step = len(train_data_loader)
    total_iterations = 0

    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths, ann_id, ans, ans_lengths) in enumerate(train_data_loader):
            # Set mini-batch dataset
            images = Variable(images)
            captions = Variable(captions)
            ans = Variable(torch.LongTensor(ans))
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                ans = ans.cuda()

            #ans, batch_sizes = pack_padded_sequence(ans, ans_lengths, batch_first=True)

            

            #inputs = Variable(images.data, volatile=True)
            #features = Variable(encoder(inputs).data)
            features = images
            #features_g, features_l = netG.encode_fc(features)
            #out = netG((features_g, features_l), y_v, lengths, state)
            features_t = encoderRNN(captions, lengths)

            features_g, features_l = netG.encode_fc(features)
            sorted_idxs = torch.cuda.LongTensor(np.argsort(ans_lengths))


            # SWEET BABY JESUS FIX THIS SHIT
            array = []
            for l in range(64):
                array.append((features_g[l], features_t[l], ans[l], ans_lengths[l], captions[l]))

            array.sort(key=lambda x: x[3], reverse=True)

            features_g, features_t, ans, ans_lengths, captions = zip(*array)

            features_g = torch.cat([ f.unsqueeze(0) for f in features_g], 0)
            features_t = torch.cat([ f.unsqueeze(0) for f in features_t], 0)
            ans        = torch.cat([ f.unsqueeze(0) for f in ans], 0)
            captions   = torch.cat([ f.unsqueeze(0) for f in captions], 0)

            targets, batch_sizes = pack_padded_sequence(ans, ans_lengths, batch_first=True)

            out = netG(features_g, features_t, ans, ans_lengths, g2_state)
            mle_loss = criterion(out, targets)
            mle_loss.backward()
            torch.nn.utils.clip_grad_norm(netG.parameters(), args.clip)
            optimizer.step()

            if total_iterations % 100 == 0:
                netG.eval()
                print("")
                #eval_input = (Variable(features_g.data, volatile=True), Variable(features_l.data, volatile=True))
                #outputs = netG(eval_input, y_v, lengths, state)
                features_g = Variable(features_g.data, volatile=True)
                features_t = Variable(features_t.data, volatile=True)

                out = netG(features_g, features_t, ans, ans_lengths, g2_state, teacher_forced=False)
                sampled_ids_free = torch.max(out,1)[1].squeeze()
                sampled_ids_free = pad_packed_sequence([sampled_ids_free, batch_sizes], batch_first=True)[0]
                sampled_ids_free = sampled_ids_free.cpu().data.numpy()

                def word_idx_to_sentence(sample, vocab, spacing=True):
                    sampled_caption = []
                    for word_id in sample:
                        word = vocab.idx2word[word_id]
                        sampled_caption.append(word)
                        if word == '<end>':
                            break
                    if spacing:
                        return ' '.join(sampled_caption)
                    else:
                        return ''.join(sampled_caption)

                groundtruth_caption = captions.cpu().data.numpy()

                ans = ans.cpu().data.numpy().tolist()
                #outputs = torch.max(outputs,1)[1]
                #outputs = outputs.cpu().data.numpy().squeeze().tolist()
                
                sampled_captions = ""
                for index in range(10):
                    if index > 1:
                        break
                    sampled_caption   = word_idx_to_sentence(groundtruth_caption[index], question_vocab)
                    gt_answer = word_idx_to_sentence(ans[index], ans_vocab,spacing=False)
                    p_answer  =  word_idx_to_sentence(sampled_ids_free[index], ans_vocab,spacing=False)
                    sampled_captions +=  "[Q]{} \n".format(sampled_caption)
                    sampled_captions += ("[A]{} \n".format(gt_answer))
                    sampled_captions += ("[P]{} \n\n".format(p_answer))

                print(sampled_captions)
                netG.train()

            # Print log info
            if total_iterations % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d] Loss: %5.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step,  mle_loss.data[0], np.exp(mle_loss.data[0])))

            # Save the model
            if (total_iterations+1) % args.save_step == 0:
                torch.save(netG.state_dict(), 
                           os.path.join(save_path, 
                                        'netG-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(save_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))


            if total_iterations % args.tb_log_step == 0:
                log_value('Loss', mle_loss.data[0], total_iterations)
                log_value('Perplexity', np.exp(mle_loss.data[0]), total_iterations)

            if (total_iterations+1) % args.save_step == 0:
                export(encoder, netG, val_data_loader, y_onehot,
                 val_state, criterion, question_vocab,ans_vocab, total_iterations, total_step, save_path)

            # if (total_iterations+1) % args.val_step == 0:
            #     validate(encoder, netG, val_data_loader, y_onehot,
            #      val_state, criterion, question_vocab,ans_vocab, total_iterations, total_step, save_path)

            total_iterations += 1


def validate(encoder, netG, val_data_loader,y_onehot, state, criterion,
 question_vocab,ans_vocab, total_iterations, total_step, save_path):
    netG.eval()
    encoder.eval()
    total_validation_loss = 0

    responses = []
    unk_idx = ans_vocab.word2idx['<unk>']
    for i, (images, captions, lengths, ann_id, ans) in enumerate(val_data_loader):
        # Set mini-batch dataset
        images = Variable(images)
        captions = Variable(captions)
        ans = Variable(torch.LongTensor(ans))
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            ans = ans.cuda()

        y_onehot.resize_(captions.size(0),captions.size(1),len(question_vocab))
        y_onehot.zero_()
        y_onehot.scatter_(2,captions.data.unsqueeze(2),1)
        y_v = Variable(y_onehot)

        netG.zero_grad()
        inputs = Variable(images.data, volatile=True)
        features = encoder(inputs)
        features_g, features_l = netG.encode_fc(features)
        outputs = netG((features_g, features_l), y_v, lengths, state, teacher_forced=True)
        mle_loss = criterion(outputs, ans)

        outputs = torch.max(outputs,1)[1]
        #outputs = torch.topk(outputs, 2, 1)[1]
        outputs = outputs.cpu().data.numpy().squeeze().tolist()

        for index in range(inputs.size(0)):
            #candidates = 
            #answer = candidates[1] if (candidates[0] == unk_idx) else candidates[0]
            answer = ans_vocab.idx2word[outputs[index]]

            responses.append({"answer":answer, "question_id": ann_id[index]})

        total_validation_loss += mle_loss.data[0]

        # Print log info
        if i % args.log_step == 0:
            print('Step [%d/%d] Val_Loss: %5.4f, Val_Perplexity: %5.4f'
                  %(i, len(val_data_loader),  mle_loss.data[0], np.exp(mle_loss.data[0])))

    average_validation_loss = total_validation_loss / len(val_data_loader)
    log_value('Val_Loss', average_validation_loss, total_iterations)

    json_save_dir = os.path.join(save_path, "{}_OpenEnded_mscoco_val2014_fake_results.json".format(total_iterations))
    json.dump(responses, open(json_save_dir, "w"))

    netG.train()
    encoder.train()
   

def export(encoder, netG, data_loader,y_onehot, state, criterion,
    question_vocab,ans_vocab, total_iterations, total_step, save_path):

    netG.eval()
    encoder.eval()

    responses = []
    for i, (images, captions, lengths, ann_id) in enumerate(data_loader):
        # Set mini-batch dataset
        images = Variable(images)
        captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        y_onehot.resize_(captions.size(0),captions.size(1),len(question_vocab))
        y_onehot.zero_()
        y_onehot.scatter_(2,captions.data.unsqueeze(2),1)
        y_v = Variable(y_onehot)

        netG.zero_grad()
        #inputs = Variable(images.data, volatile=True)
        #features = encoder(inputs)
        features = images
        features_g, features_l = netG.encode_fc(features)
        outputs = netG((features_g, features_l), y_v, lengths, state, teacher_forced=True)
        outputs = torch.max(outputs,1)[1]
        outputs = outputs.cpu().data.numpy().squeeze().tolist()

        for index in range(images.size(0)):
            answer = ans_vocab.idx2word[outputs[index]]
            responses.append({"answer":answer, "question_id": ann_id[index]})

        # Print log info
        if i % args.log_step == 0:
            print('Step [%d/%d] Exporting  ... '
                  %(i, len(data_loader)))

    json_save_dir = os.path.join(save_path, "{}_OpenEnded_mscoco_val2014_fake_results.json".format(total_iterations))
    json.dump(responses, open(json_save_dir, "w"))

    dataDir = 'data'

    taskType    ='OpenEnded'
    dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
    dataSubType ='val2014'
    annFile     ='%s/Annotations/v2_%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
    quesFile    ='%s/Questions/v2_%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
    imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

    resFile = json_save_dir

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()

    print "\n"
    print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
    log_value('Val_Acc', vqaEval.accuracy['overall'], total_iterations)

    netG.train()
    encoder.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=299 ,
                        help='image size to use')
    parser.add_argument('--question_vocab_path', type=str, default='data/question_vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--ans_vocab_path', type=str, default='data/ans_vocab_char.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--dataset', type=str, default='coco' ,
                        help='dataset to use')
    parser.add_argument('--comments_path', type=str,
                        default='data/labels.h5',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
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
    parser.add_argument('--netG', type=str)
    parser.add_argument('--encoder', type=str)
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
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

    configure(save_path)
    run(save_path, args)