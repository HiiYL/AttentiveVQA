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

        self.embed = nn.Sequential(
            nn.Linear(self.question_vocab_size, embed_size),
            nn.Dropout(0.5),
            nn.ReLU()
        )

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
        if isinstance(captions, tuple):
            captions, batch_sizes = captions
            embeddings = self.embed(captions)
            embeddings = pad_packed_sequence((embeddings,batch_sizes), batch_first=True)[0]
        else:
            batch_size, caption_length, _ = captions.size()
            captions = captions.view(-1,self.question_vocab_size)
            embeddings = self.embed(captions)
            embeddings = embeddings.view(batch_size, caption_length, -1)

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

        return out


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




