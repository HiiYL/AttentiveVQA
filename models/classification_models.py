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
from LSTM import LSTMAttentive, LSTMSimple, LSTMSpatial
from encoder import EncoderFC

class G_Spatial(nn.Module):
    def __init__(self, embed_size, hidden_size, question_vocab, ans_vocab, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(G_Spatial, self).__init__()
        self.encode_fc = EncoderFC()
        self.vocab = question_vocab

        self.vocab_size = len(question_vocab)
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embed = nn.Sequential(
            nn.Linear(len(question_vocab), embed_size),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.lstm    = LSTMSimple(embed_size, hidden_size)
        self.lstm_2  = LSTMSimple(embed_size, hidden_size)

        self.fc  =  nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(hidden_size * 2, len(ans_vocab))
        )

        self.log_softmax = nn.LogSoftmax()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

         
    def forward(self, features, captions, lengths, states, skip=2, returnHidden=False):
        """Decode image feature vectors and generates captions."""
        features_global, features_local = features
        if isinstance(captions, tuple):
            captions, batch_sizes = captions
            embeddings = self.embed(captions)
            embeddings = pad_packed_sequence((embeddings,batch_sizes), batch_first=True)[0]
        else:
            batch_size, caption_length, _ = captions.size()
            captions = captions.view(-1,self.vocab_size)
            embeddings = self.embed(captions)
            embeddings = embeddings.view(batch_size, caption_length, -1)

        embeddings = torch.cat((features_global.unsqueeze(1), embeddings), 1)
        batch_size = features_global.size(0)

        hiddens_tensor = Variable(torch.cuda.FloatTensor(len(lengths),lengths[0],self.hidden_size * 2))
        context_tensor = Variable(torch.cuda.FloatTensor(len(lengths),lengths[0],self.hidden_size * 2))
        hx, cx = states
        hx, cx = hx[0], cx[0]

        if hx.size(0) != batch_size:
            hx = hx[:batch_size]
            cx = cx[:batch_size]

        j = lengths[0]
        for i in range(lengths[0]):
            inputs          = embeddings[:,i,:]
            inputs_reversed = embeddings[:, j-i, :]

            #inputs          = torch.cat((inputs, features_global),1)
            #inputs_reversed = torch.cat((inputs_reversed, features_global),1)

            #inputs          = self.visual_embed(inputs)
            #inputs_reversed = self.visual_embed(inputs_reversed)

            hy, cy = self.lstm( inputs, hx, cx, features_local)
            hx, cx = self.lstm_2( inputs_reversed, hy, cy, features_local)

            # if i >= skip and (i % (skip + 1) == 0):
            #     hx = hx + hiddens_tensor[:, i - skip, :]
            #     cx = cx + context_tensor[:, i - skip, :]
            hiddens_tensor[ :, i, :] = torch.cat((hx, hy), 1)
            context_tensor[ :, i, :] = torch.cat((cx, cy), 1)

        #(64L, 16L, 1024L)

        results_tensor = hiddens_tensor + context_tensor #torch.cat((hiddens_tensor, context_tensor), 2)

        if returnHidden:
            return results_tensor
        else:
            combined = [ results_tensor[i, length - 1, :].unsqueeze(0) for i, length in enumerate(lengths)]
            combined = torch.cat(combined, 0)
            outputs = self.fc(combined)
            return outputs


