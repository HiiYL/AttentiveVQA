import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,PackedSequence
import torchvision.models as models
import os
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform
from LSTM import LSTMAttentive, LSTMSimple, LSTMSpatial, LSTMCustom
from encoder import EncoderFC

class MultimodalAttentionRNN(nn.Module):
    def __init__(self, len_vocab, glimpse=2):
        super(MultimodalAttentionRNN, self).__init__()
        self.block1 = MLBBlock(2400, 1200, glimpse)
        self.block2 = MLBBlockReversed(1200 * glimpse, 1200, glimpse)
        #self.block3 = MLBBlock(1200 * glimpse, 1200, glimpse)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1200 * glimpse, 1200 * glimpse),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(1200 * glimpse, len_vocab)
        )

        #self.classifier_aux = nn.Linear(1200 * glimpse, 1)

        # self.classifier_qtype = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(1200 * glimpse, 1200 * glimpse),
        #     nn.Tanh(),
        #     nn.Dropout(),
        #     nn.Linear(1200 * glimpse, 65)
        # )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self,v,q, q_full, lengths):
        batch_size = q.size(0)

        V = self.block1(v,q)

        v_gap = F.avg_pool2d(v, kernel_size=v.size()[2:])
        Q = self.block2(v_gap, q_full, lengths)

        z = V * Q
        # y = self.block2(v,x)
        # z = self.block3(v,y)
        # x = x + z
        out = self.classifier(z)

        # if self.training:
        #     #out_type = self.classifier_qtype(x)
        #     out_confidence = self.classifier_aux(x)
        #     return out, out_confidence
        # else:
        #     return out
        return out

class MLBBlock(nn.Module):
    def __init__(self, input_size, output_size, glimpse):
        super(MLBBlock, self).__init__()
        self.glimpse = glimpse
        self.visual_attn = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(2048, 1200, kernel_size=1,stride=1,padding=0),
            nn.Tanh()
        )
        self.question_attn = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size,output_size),
            nn.Tanh(),
        )

        self.attn_block = nn.Conv2d(1200, glimpse, kernel_size=1,stride=1,padding=0)

        # visual_embed = []
        # for i in range(glimpse):
        #     visual_embed.append(
        #         nn.Sequential(
        #             nn.Dropout(),
        #             nn.Linear(2048,output_size),
        #             nn.Tanh()
        #         )
        #     )
        # self.visual_embed = ListModule(*visual_embed)

        self.visual_embed = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048 * glimpse, output_size * glimpse),
            nn.Tanh()
        )
        # self.question_block = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(input_size,output_size * glimpse),
        #     nn.Tanh(),
        # )
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self,v, q):
        v_atten  = self.visual_attn(v)
        batch_size, depth, height, width = v_atten.size()
        v_atten     = v_atten.view(batch_size, depth, -1)
        # 200 x 1200 x 64
        q_atten  = self.question_attn(q)
        q_replicate = q_atten.unsqueeze(2).expand_as(v_atten)
        # 200 x 1200 x 64

        attn = v_atten * q_replicate
        attn = attn.view(batch_size, depth, height, width)
        attn = self.attn_block(attn)
        attn = attn.view(attn.size(0),attn.size(1), -1)
        # 200 x glimpse x 64
        
        # apply softmax along each glimpse slice
        # b - batch
        # s - glimpse_slice
        # d - depth ( spatial features )
        b, s, d = attn.size()
        attn = F.softmax(attn.view(b * s, d)).view(b, s, d)
        # 200 x glimpse x 64 

        v    = v.view(v.size(0), v.size(1), -1)
        # 200 x 2048 x 64
        v    = v.permute(0,2,1).contiguous()
        # 200 x 64 x 2048 

        v    = torch.bmm(attn, v)
        # 200 x glimpse x 2048

        b, s, d = v.size()
        #V = torch.cat([ self.visual_embed[i](v[:,i,:]) for i in range(s) ], 1)
        V = self.visual_embed(v.view(b, -1))
        # 200 x glimpse * 1200
        #Q = self.question_block(q)

        #out = V * Q

        return V #out

class MLBBlockReversed(nn.Module):
    def __init__(self, input_size, output_size, glimpse):
        super(MLBBlockReversed, self).__init__()
        self.glimpse = glimpse
        self.visual_attn = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(2048, 1200, kernel_size=1,stride=1,padding=0),
            nn.Tanh()
        )
        self.question_attn = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size,output_size),
            nn.Tanh(),
        )

        self.attn_block = nn.Conv1d(1200, glimpse, kernel_size=1, stride=1)#nn.Conv2d(1200, glimpse, kernel_size=1,stride=1,padding=0)

        # self.visual_embed = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(2048, output_size * glimpse),
        #     nn.Tanh()
        # )
        self.question_block = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size * glimpse,output_size * glimpse),
            nn.Tanh(),
        )
        self.adaptive_q_pool = nn.AdaptiveAvgPool1d(20) #nn.AdaptiveAvgPool2d((20,input_size))#
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self,v, q, lengths):
        v_atten  = self.visual_attn(v)
        batch_size, depth, height, width = v_atten.size()
        v_atten     = v_atten.view(batch_size, 1, depth)
        q = q.permute(0,2,1)
        q = self.adaptive_q_pool(q)
        q = q.permute(0,2,1).contiguous()
        # 100 x 20 x 2400
        b, s, d = q.size()
        q_atten  = self.question_attn(q.view(b * s, d)).view(b,s,-1)
        # 100 x 20 x 1200

        v_replicate = v_atten.expand_as(q_atten)
        # 100 x 20 x 1200

        attn = q_atten * v_replicate
        attn = attn.permute(0,2,1).contiguous()
        attn = self.attn_block(attn)
        # 200 x glimpse x 64
        
        # apply softmax along each glimpse slice
        # b - batch
        # s - glimpse_slice
        # d - depth ( spatial features )
        b, s, d = attn.size()
        attn = F.softmax(attn.view(b * s, d)).view(b, s, d)

        # attn - 100 x 2 x 20 | q - 100 x 20 x 2400
        q    = torch.bmm(attn, q)
        # 200 x glimpse x 2048
        #V = self.visual_embed(v.view(v.size(0), -1))
        # 200 x glimpse * 1200
        Q = self.question_block(q.view(q.size(0), -1))
        #out = V * Q

        return Q





class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)