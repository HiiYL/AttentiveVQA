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

class EncoderCNN(nn.Module):
    def __init__(self,inception, requires_grad=False):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.aux_logits = False
        self.transform_input = False
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        # if aux_logits:
        #     self.AuxLogits = inception.AuxLogits
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c

        for parameters in self.parameters():
            parameters.requires_grad = requires_grad

    def forward(self, x):
        """Extract the image feature vectors."""
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        # if self.training and self.aux_logits:
        #     aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        #x = F.dropout(x, training=self.training)

        return x

# class EncoderFC(nn.Module):
#     def __init__(self, global_only=False):
#         super(EncoderFC, self).__init__()
#         self.global_only = global_only
#         self.fc_global = nn.Sequential(
#             nn.Conv2d(2048, 512, kernel_size=1,stride=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )
#         if not global_only:
#             self.fc_local = nn.Sequential(
#                 nn.Conv2d(2048, 512, kernel_size=1,stride=1, padding=0),
#                 nn.BatchNorm2d(512),
#                 nn.ReLU()
#             )
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 kaiming_uniform(m.weight.data)
#                 m.bias.data.zero_()

#     def forward(self, x):
#         x_global = self.fc_global(x)
#         x_global = F.avg_pool2d(x_global, kernel_size=x_global.size()[2:])[:,:,0,0]

#         if not self.global_only:
#             x_local = self.fc_local(x)
#             x_local = x_local.view(x_local.size(0), x_local.size(1), x_local.size(2) * x_local.size(3))
#             return x_global, x_local

#         return x_global

# class EncoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
#         """Set the hyper-parameters and build the layers."""
#         super(EncoderRNN, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#         self.init_weights()
    
#     def init_weights(self):
#         """Initialize weights."""
#         self.embed.weight.data.uniform_(-0.1, 0.1)
        
#     def forward(self, captions, lengths):
#         """Decode image feature vectors and generates captions."""
#         embeddings = self.embed(captions)
#         packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
#         hiddens, _ = self.lstm(packed)

#         hiddens_padded, _ = pad_packed_sequence(hiddens, batch_first=True)
#         last_hiddens = [ hiddens_padded[i, length - 1, :].unsqueeze(0) for i, length in enumerate(lengths)]
#         last_hiddens = torch.cat(last_hiddens, 0)
#         return last_hiddens #hiddens_padded


from skipthought.skipthoughts import UniSkip, BiSkip,BayesianUniSkip
class EncoderSkipThought(nn.Module):
    def __init__(self, vocab):
        """Set the hyper-parameters and build the layers."""
        super(EncoderSkipThought, self).__init__()

        dir_st = 'skipthought/data/skip-thoughts'
        self.buskip = BayesianUniSkip(dir_st, vocab.word2idx.keys(), dropout=.25)

    def forward(self, inputs, lengths):
        return self.buskip(inputs, lengths=lengths)
