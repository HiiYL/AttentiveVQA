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
        #self.block2 = MLBBlock(1200 * glimpse, 1200, glimpse)
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

    def forward(self,v,q):

        for i in range(3):
            x = self.block1(v,q)
        #x = self.block2(v,x)
        #x = self.block3(v,x)
        out = self.classifier(x)

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
        self.question_block = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size,output_size * glimpse),
            nn.Tanh(),
        )
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
        Q = self.question_block(q)

        out = V * Q

        return out


class G_Spatial(nn.Module):
    def __init__(self, embed_size, hidden_size, question_vocab, ans_vocab, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(G_Spatial, self).__init__()
        self.encode_fc = EncoderFC()

        self.question_vocab = question_vocab
        self.ans_vocab      = ans_vocab

        self.question_vocab_size = len(question_vocab)
        self.ans_vocab_size      = len(ans_vocab)

        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embed = nn.Embedding(self.question_vocab_size, embed_size)

        self.lstm    = LSTMCustom(embed_size, hidden_size)
        #self.lstm2   = LSTMCustom(embed_size, hidden_size)

        self.fc  =  nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, self.ans_vocab_size)
        )

        self.log_softmax = nn.LogSoftmax()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

         
    def forward(self, features, captions, lengths, states):
        """Decode image feature vectors and generates captions."""
        features_global, features_local = self.encode_fc(features)
        embeddings = self.embed(captions)

        embeddings = torch.cat((features_global.unsqueeze(1), embeddings), 1)
        batch_size = features_global.size(0)

        hiddens_tensor = Variable(torch.cuda.FloatTensor(len(lengths),lengths[0],self.hidden_size))
        context_tensor = Variable(torch.cuda.FloatTensor(len(lengths),lengths[0],self.hidden_size))
        hx, cx = states
        hx, cx = hx[0], cx[0]

        if hx.size(0) != batch_size:
            hx = hx[:batch_size]
            cx = cx[:batch_size]

        #j = lengths[0]
        # features_global_1 = self.lstm.visual_transform(features_global)
        # features_global_2 = self.lstm2.visual_transform(features_global)
        for i in range(lengths[0]):
            inputs          = embeddings[:,i,:]
            #inputs_reversed = embeddings[:, j-i, :]

            hx, cx = self.lstm( inputs, hx, cx, features_local, features_global)
            #hx, cx = self.lstm2( inputs, hx, cx, features_local, features_global)
            #hx, cx = self.lstm_2( inputs_reversed, hy, cy, features_local)

            hiddens_tensor[ :, i, :] = hx#torch.cat((hx, hy), 1)
            context_tensor[ :, i, :] = cx#torch.cat((cx, cy), 1)

        results_tensor = hiddens_tensor + context_tensor #torch.cat((hiddens_tensor, context_tensor), 2)
        combined = [ results_tensor[i, length - 1, :].unsqueeze(0) for i, length in enumerate(lengths)]
        combined = torch.cat(combined, 0)
        outputs = self.fc(combined)
        return outputs


class MultimodalRNN(nn.Module):
    def __init__(self, len_vocab):
        super(MultimodalRNN, self).__init__()
        self.block1 = MultimodalBlock(2400, 1200)
        self.block2 = MultimodalBlock(1200, 1200)
        self.block3 = MultimodalBlock(1200, 1200)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1200, len_vocab)
        )
        
        self.classifier_qtype = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1200, 65)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-0.08, 0.08)#kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self,v,q):
        v = F.avg_pool2d(v, kernel_size=v.size()[2:])[:,:,0,0]

        h1 = self.block1(v,q)
        h2 = self.block2(v,h1)
        h3 = self.block3(v,h2)

        out = self.classifier(h3)
        out_type = self.classifier_qtype(h3)

        return out, out_type


class MultimodalBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultimodalBlock, self).__init__()
        self.visual_block = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048,2048),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(2048,output_size),
            nn.Tanh(),
            
        )
        self.question_block = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size,output_size),
            nn.Tanh(),
        )
        self.linear = nn.Linear(input_size,output_size)
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-0.08, 0.08)#kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self,v, q):
        V = self.visual_block(v)
        Q = self.question_block(q)

        combined = V * Q

        q_skip = self.linear(q)
        out = q_skip + combined

        return out




class MultimodalAttentionBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultimodalAttentionBlock, self).__init__()
        self.visual_block = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048,2048),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(2048,output_size),
            nn.Tanh(),
            
        )
        self.attn_block = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size,64),
            nn.Softmax(),
        )
        self.question_block = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size,output_size),
            nn.Tanh(),
        )
        self.linear = nn.Linear(input_size,output_size)
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-0.08, 0.08)#kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self,v, q):
        attn = self.attn_block(q)
        v = torch.bmm(v.view(v.size(0), v.size(1), -1), attn.unsqueeze(2)).squeeze(2)

        V = self.visual_block(v)
        Q = self.question_block(q)

        combined = V * Q

        q_skip = self.linear(q)
        out = q_skip + combined

        return out





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