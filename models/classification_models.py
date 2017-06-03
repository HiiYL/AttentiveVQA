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
    def __init__(self, ans_vocab, glimpse=2):
        super(MultimodalAttentionRNN, self).__init__()
        self.blockv = MLBBlockVAttention(2400, 1200, glimpse, joint=False)
        #self.blockq = MLBBlockQAttention(2048, 1200, glimpse, joint=True)
        self.classifier = nn.Sequential(
             nn.Dropout(),
             nn.Linear(1200 * glimpse, len(ans_vocab))
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self,v,q, q_full, lengths):
        z = self.blockv(v,q) #* self.blockq(v, q_full, lengths)
        out = self.classifier(z)

        return out

class MLBBlockVAttention(nn.Module):
    def __init__(self, input_size, output_size, glimpse, joint=False):
        super(MLBBlockVAttention, self).__init__()
        self.glimpse = glimpse
        self.visual_attn = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(2048, output_size, kernel_size=1,stride=1,padding=0),
            nn.PReLU()
        )
        self.question_attn = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size,output_size),
            nn.PReLU()
        )
        self.attn_block = nn.Conv2d(1200, glimpse, kernel_size=1,stride=1,padding=0)
        self.visual_embed = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048 * glimpse, output_size * glimpse),
            nn.PReLU()
        )
        self.joint = joint
        if not joint:
            self.question_embed = nn.Sequential(
                nn.Dropout(),
                nn.Linear(2400, output_size * glimpse),
                nn.PReLU()
            )
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self,v, q):
        v_atten  = self.visual_attn(v)
        batch_size, depth, height, width = v_atten.size()
        v_atten     = v_atten.view(batch_size, depth, -1)
        # 200 x 1200 x 64
        q_atten  = self.question_attn(q).unsqueeze(2)
        q_replicate = q_atten.expand_as(v_atten)
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
        V = self.visual_embed(v.view(v.size(0), -1))

        if self.joint:
            return V
        else:
            q = self.question_embed(q)
            return V * q

class MLBBlockQAttention(nn.Module):
    def __init__(self, input_size, output_size, glimpse, joint=False):
        super(MLBBlockQAttention, self).__init__()
        self.glimpse = glimpse
        self.adaptive_q_pool = nn.AdaptiveAvgPool1d(64)
        self.visual_attn = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, output_size),
            nn.PReLU()
        )
        self.question_attn = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2400,output_size),
            nn.PReLU()
        )
        self.attn_block = nn.Conv1d(1200, glimpse, kernel_size=1, stride=1)
        self.question_embed = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2400 * glimpse,output_size * glimpse),
            nn.PReLU()
        )
        self.joint = joint
        if not joint:
            self.visual_embed = nn.Sequential(
                nn.Dropout(),
                nn.Linear(2048, output_size * glimpse),
                nn.PReLU()
            )
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self,v, q, lengths):
        v       = F.avg_pool2d(v, kernel_size=v.size()[2:])
        v       = v.view(v.size(0), v.size(1))
        v_atten = self.visual_attn(v).unsqueeze(1)

        q = q.permute(0,2,1)
        q = self.adaptive_q_pool(q)
        q = q.permute(0,2,1).contiguous()

        # 100 x 64 x 2400
        b, s, d = q.size()
        q_atten  = self.question_attn(q.view(b * s, d)).view(b,s,-1)
        # 100 x 64 x 1200
        v_replicate = v_atten.expand_as(q_atten)
        # 100 x 64 x 1200

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

        q    = torch.bmm(attn, q)
        # 200 x glimpse x 2048

        Q = self.question_embed(q.view(q.size(0), -1))

        if self.joint:
            return Q
        else:
            v = self.visual_embed(v)
            return Q * v