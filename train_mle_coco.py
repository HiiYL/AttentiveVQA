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
from models_spatial import EncoderCNN, G_Spatial
import pickle
import datetime

import json

from tensorboard_logger import configure, log_value

def run(save_path, args):
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
    
    train_data_loader = get_loader("train", question_vocab, ans_vocab,
                             train_transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    # val_data_loader = get_loader("val", vocab, 
    #                          test_transform, 1,
    #                          shuffle=False, num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size,models.inception_v3(pretrained=True), requires_grad=False)
    netG = G_Spatial(args.embed_size, args.hidden_size, question_vocab, ans_vocab, args.num_layers)

    if args.netG:
        print("[!]loading pretrained netG....")
        netG.load_state_dict(torch.load(args.netG))
        print("Done!")

    if args.encoder:
        print("[!]loading pretrained decoder....")
        encoder.load_state_dict(torch.load(args.encoder))
        print("Done!")

    criterion = nn.CrossEntropyLoss()
    
    state = (Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)),
     Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)))
    val_state = (Variable(torch.zeros(args.num_layers, 1, args.hidden_size)),
     Variable(torch.zeros(args.num_layers, 1, args.hidden_size)))

    y_onehot = torch.FloatTensor(args.batch_size, 20,len(question_vocab))

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        netG.cuda()
        #netD.cuda()
        state = [s.cuda() for s in state]
        val_state = [s.cuda() for s in val_state]
        criterion = criterion.cuda()
        y_onehot = y_onehot.cuda()

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
        for i, (images, captions, lengths, img_id, ans) in enumerate(train_data_loader):
            # Set mini-batch dataset
            images = Variable(images)
            captions = Variable(captions)
            ans = Variable(torch.LongTensor(ans))
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                ans = ans.cuda()

            #ans, batch_sizes = pack_padded_sequence(ans, ans_lengths, batch_first=True)

            y_onehot.resize_(captions.size(0),captions.size(1),len(question_vocab))
            y_onehot.zero_()
            y_onehot.scatter_(2,captions.data.unsqueeze(2),1)
            y_v = Variable(y_onehot)

            netG.zero_grad()
            inputs = Variable(images.data, volatile=True)
            features = Variable(encoder(inputs).data)
            features_g, features_l = netG.encode_fc(features)
            out = netG((features_g, features_l), y_v, lengths, state, teacher_forced=True)

            mle_loss = criterion(out, ans)
            mle_loss.backward()
            torch.nn.utils.clip_grad_norm(netG.parameters(), args.clip)
            optimizer.step()


            if total_iterations % 100 == 0:
                netG.eval()
                print("")
                eval_input = (Variable(features_g.data, volatile=True), Variable(features_l.data, volatile=True))
                outputs = netG(eval_input, y_v, lengths, state, teacher_forced=True)

                def word_idx_to_sentence(sample):
                    sampled_caption = []
                    for word_id in sample:
                        word = question_vocab.idx2word[word_id]
                        sampled_caption.append(word)
                        if word == '<end>':
                            break
                    return ' '.join(sampled_caption)

                groundtruth_caption = captions.cpu().data.numpy()

                ans = ans.cpu().data.numpy().tolist()
                outputs = torch.max(outputs,1)[1]
                outputs = outputs.cpu().data.numpy().squeeze().tolist()
                
                sampled_captions = ""
                for i in range(10):
                    if i > 1:
                        break
                    sampled_caption   = word_idx_to_sentence(groundtruth_caption[i])
                    sampled_captions +=  "[Q]{} \n".format(sampled_caption)
                    sampled_captions += ("[A]{} \n".format(ans_vocab.idx2word[ans[i]]))
                    sampled_captions += ("[P]{} \n\n".format(ans_vocab.idx2word[outputs[i]]))

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

            # if (total_iterations+1) % args.save_step == 0:
            #     validate(encoder, netG, val_data_loader, val_state, criterion, vocab, total_iterations)

            total_iterations += 1



# def validate(encoder, netG, val_data_loader, state, criterion, vocab, total_iterations):
#     ### MS COCO Eval code prepation
#     ## set up file names and pathes
#     dataDir='data/coco'
#     logDir=save_path
#     dataType='val2014'
#     algName = 'fakecap'
#     annFile='%s/captions_%s_karpathy_split.json'%(dataDir,dataType)
#     subtypes=['results', 'evalImgs', 'eval']
#     [resFile, evalImgsFile, evalFile]= \
#     ['%s/captions_%s_%s_%s.json'%(logDir,dataType,algName,subtype) for subtype in subtypes]
#     ###


#     print('validating...')
#     netG.eval()
#     encoder.eval()

#     for parameter in encoder.parameters():
#         parameter.requires_grad=False
#     for parameter in netG.parameters():
#         parameter.requires_grad=False



#     val_json = []
#     total_val_step = len(val_data_loader)
#     for i, (images, captions, lengths, img_id) in enumerate(val_data_loader):
#         images = Variable(images, volatile=True)
#         captions = Variable(captions, volatile=True)
#         if torch.cuda.is_available():
#             images = images.cuda()
#             captions = captions.cuda()
#         targets, batch_sizes = pack_padded_sequence(captions, lengths, batch_first=True)

#         # Forward, Backward and Optimize
#         features = encoder(images)
#         features_g, features_l = netG.encode_fc(features)

#         sampled_ids_list, terminated_confidences = netG.beamSearch((features_g, features_l), state, n=3, diverse_gamma=0.0)
#         sampled_ids_list = np.array([ sampled_ids.cpu().data.numpy() for sampled_ids in sampled_ids_list ])

#         max_index = np.argmax(terminated_confidences)
#         #print(max_index)
#         sampled_ids = sampled_ids_list[max_index]

#         sampled_caption = []
#         for word_id in sampled_ids:
#             word = vocab.idx2word[word_id]
#             if word == '<end>' or word == '<pad>':
#                 break
#             if word != '<start>':
#                 sampled_caption.append(word)

#         sentence = ' '.join(sampled_caption)
#         item_json = {"image_id": img_id[0], "caption": sentence}
#         val_json.append(item_json)

#         # # Print log info
#         if i % (args.log_step * 10) == 0:
#            print('[%d/%d] - Running model on validation set....'
#                  %(i, total_val_step))


#     path, file = os.path.split(resFile)

#     export_filename = '{}/{}_{}'.format(path, total_iterations, file)
#     print("Exporting to... {}".format(export_filename))
#     with open(export_filename, 'w') as outfile:
#         json.dump(val_json, outfile)

#     print("calculating metrics...")
#     coco = COCO(annFile)
#     cocoRes = coco.loadRes(export_filename)
#     cocoEval = COCOEvalCap(coco, cocoRes)
#     cocoEval.params['image_id'] = cocoRes.getImgIds()
#     cocoEval.evaluate()

#     for metric, score in cocoEval.eval.items():
#         log_value(metric, score, total_iterations)
#         print '%s: %.3f'%(metric, score)

#     netG.train()
#     encoder.train()
#     for parameter in encoder.parameters():
#         parameter.requires_grad=True
#     for parameter in netG.parameters():
#         parameter.requires_grad=True
                
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
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--tb_log_step', type=int , default=100,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=8856,
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
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
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