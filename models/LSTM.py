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

class LSTMAttentive(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(LSTMAttentive, self).__init__()

        self.sentinel_igate = nn.Linear(hidden_size, hidden_size)
        self.sentinel_hgate = nn.Linear(hidden_size, hidden_size)

        self.sentinel_transform_gate = nn.Linear(hidden_size, hidden_size)
        self.hidden_transform_gate   = nn.Linear(hidden_size, hidden_size)

        self.v_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.5)
        )

        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size)

        self.attn_sentinel = nn.Linear( hidden_size, 1)
        self.attn_visual   = nn.Linear( hidden_size, 1)

        self._initialize_weights()

    def applyLinearOn3DMatrix(self, layer_input, layer):
        batch_size, timestep, size = layer_input.size()
        out = layer(layer_input.view(-1, size))
        out = out.view(batch_size, timestep, -1)
        return out

    def forward(self, inputs, hx, cx, features):
        hy, cy = self.lstm_cell(inputs, (hx,cx))

        sentinel_i, sentinel_h = self.sentinel_igate(inputs), self.sentinel_hgate(hx)
        sentinel_gate = F.sigmoid(sentinel_i + sentinel_h)
        sentinel = sentinel_gate * F.tanh(cy)

        # Add sentinel column to feature
        features_with_sentinel = torch.cat((features, sentinel.unsqueeze(1)),1)
        # Calculate attention with sentinel
        h_t   = self.hidden_transform_gate(hy)
        s_t   = self.sentinel_transform_gate(sentinel)
        z_h_t = h_t.unsqueeze(1).expand_as(features)

        v    = self.applyLinearOn3DMatrix(features, self.v_transform)
        z_t  = F.tanh( v   + z_h_t )
        st_t = F.tanh( h_t + s_t   )

        base_atten     = self.applyLinearOn3DMatrix(z_t, self.attn_visual ).squeeze(2)
        sentinel_atten = self.attn_sentinel(st_t)

        attn_weights = F.softmax(torch.cat((base_atten, sentinel_atten), 1))
        visual_cy    = torch.bmm(attn_weights.unsqueeze(1), features_with_sentinel).squeeze(1)

        beta   = attn_weights[:,-1].unsqueeze(1).expand_as(sentinel)
        cy_hat = beta * sentinel + (1 - beta) * visual_cy
        
        return hy, cy_hat


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

class LSTMSpatial(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(LSTMSpatial, self).__init__()
        self.hidden_transform  = nn.Linear(hidden_size, hidden_size)

        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size)
        self.attn = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def applyLinearOn3DMatrix(self, layer_input, layer):
        batch_size, timestep, size = layer_input.size()
        out = layer(layer_input.view(-1, size))
        out = out.view(batch_size, timestep, -1)
        return out

    def forward(self, inputs, hx, cx, features):
        batch_size, d, k  = features.size()
        hy, cy = self.lstm_cell(inputs, (hx,cx))

        Wh = self.hidden_transform(hy)
        Wh = Wh.unsqueeze(2).expand_as(features)

        attn_in = F.tanh(features + Wh).view(batch_size, d, 8, 8)
        z_t = self.attn(attn_in).view(batch_size, 64)
        attn_weights = F.softmax(z_t)
        cy    = torch.bmm(features, attn_weights.unsqueeze(2)).squeeze(2)

        return hy, cy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

class LSTMCustom(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(LSTMCustom, self).__init__()
        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size)

        self.sentinel_igate = nn.Linear(hidden_size, hidden_size)
        self.sentinel_hgate = nn.Linear(hidden_size, hidden_size)

        self.attn      = nn.Linear( hidden_size * 2, 64)
        self.attn_sentinel  = nn.Linear( hidden_size, 1)

        self._initialize_weights()


    def forward(self, inputs, hx, cx, features_spatial, features_global):
        hy, cy = self.lstm_cell(inputs, (hx,cx))

        sentinel_i, sentinel_h = self.sentinel_igate(inputs), self.sentinel_hgate(hx)
        sentinel_gate          = F.sigmoid(sentinel_i + sentinel_h)
        sentinel               = sentinel_gate * F.tanh(cy)

        features_spatial       = torch.cat((features_spatial, sentinel.unsqueeze(2)),2)
        base_atten     = self.attn(torch.cat((inputs, hy), 1))
        sentinel_atten = self.attn_sentinel(sentinel)
        
        attn_weights = F.softmax(torch.cat((base_atten, sentinel_atten), 1))
        visual_cy    = torch.bmm(features_spatial, attn_weights.unsqueeze(2)).squeeze(2)

        beta   = attn_weights[:,-1].unsqueeze(1).expand_as(sentinel)
        cy     = cy + ( beta * sentinel + ( 1 - beta ) * visual_cy )

        hy = hy * features_global

        return hy, cy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

class LSTMSimple(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(LSTMSimple, self).__init__()
        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size)
        self.attn = nn.Linear( hidden_size * 2, 64)

        self._initialize_weights()

    def forward(self, inputs, hx, cx, features):

        hy, cy = self.lstm_cell(inputs, (hx,cx))

        attn_weights = F.softmax(self.attn(torch.cat((inputs, hy), 1)))
        visual_cy    = torch.bmm(features, attn_weights.unsqueeze(2)).squeeze(2)

        cy = visual_cy + cy
        return hy, cy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()


