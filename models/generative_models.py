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
from LSTM import LSTMAttentive, LSTMSimple

from encoder import EncoderFC


class LSTMSimple(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(LSTMSimple, self).__init__()
        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size)
        self.attn = nn.Linear( hidden_size * 2, 64)

        self._initialize_weights()

    def forward(self, inputs, hx, cx):
        hy, cy = self.lstm_cell(inputs, (hx,cx))

        #attn_weights = F.softmax(self.attn(torch.cat((inputs, hy), 1)))
        #visual_cy    = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)

        #cy = visual_cy + cy
        
        return hy, cy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

class G_Spatial_Adversarial(nn.Module):
    def __init__(self, embed_size, hidden_size, question_vocab, ans_vocab, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(G_Spatial_Adversarial, self).__init__()

        self.encode_fc = EncoderFC()

        # self.fc = nn.Sequential (
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(4096, len(ans_vocab))
        # )

        self.vocab_size = len(ans_vocab)
        self.text_adaptive = nn.AdaptiveAvgPool1d(20)

        self.hidden_size = hidden_size * 4
        self.embed_size = embed_size * 4

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        #self.lstm = nn.LSTM(embed_size * 4, hidden_size * 4, num_layers, batch_first=True)

        self.lstm = LSTMSimple(self.embed_size, self.hidden_size)

        self.linear = nn.Linear(self.hidden_size * 2, self.vocab_size)


        # self.vocab = question_vocab

        # self.vocab_size = len(question_vocab)
        # self.hidden_size = hidden_size
        # self.embed_size = embed_size

        # self.lstm  = LSTMSimple(embed_size, hidden_size)

        # self.fc  = nn.Sequential(
        #     nn.Linear(hidden_size * 2,hidden_size * 2),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(hidden_size * 2,hidden_size * 2),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(hidden_size * 2, len(ans_vocab))
        # )
        # self.log_softmax = nn.LogSoftmax()
        # self.dropout = nn.Dropout()
        # self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self, image_features, text_features, answer, answer_lengths, states, skip=2, teacher_forced=True):
        """Decode image feature vectors and generates captions."""
        if teacher_forced:
            return self._forward_forced_cell(image_features, text_features, answer, answer_lengths, states, skip=2)
        else:
            return self._forward_free_cell(image_features, text_features, answer_lengths, states, skip=2)

    def _forward_forced_cell(self, image_features, text_features, answer, answer_lengths, states, skip=2):
        #text_features_pooled = self.text_adaptive(text_features.permute(0,2,1)).permute(0,2,1)
        embeddings = self.embed(answer)
        combined = torch.cat((image_features, text_features), 1).unsqueeze(1)

        max_length = max(answer_lengths)
        hiddens_tensor = Variable(torch.cuda.FloatTensor(len(answer_lengths),max_length,self.hidden_size))
        context_tensor = Variable(torch.cuda.FloatTensor(len(answer_lengths),max_length,self.hidden_size))
        hx, cx = states
        hx, cx = hx[0], cx[0]

        batch_size = image_features.size(0)
        if hx.size(0) != batch_size:
            hx = hx[:batch_size]
            cx = cx[:batch_size]

        embeddings = torch.cat((combined, embeddings), 1)
        for i in range(max_length):
            inputs = embeddings[:,i,:]
            hx, cx = self.lstm( inputs, hx, cx)#, features_local)

            hiddens_tensor[ :, i, :] = hx
            context_tensor[ :, i, :] = cx

            if i >= skip and (i % (skip + 1) == 0):
                hx = hx + hiddens_tensor[:, i - skip, :]
                cx = cx + context_tensor[:, i - skip, :]


        results_tensor = torch.cat((hiddens_tensor, context_tensor), 2)

        #combined = [ results_tensor[i, answer_lengths - 1, :].unsqueeze(0) for i, answer_lengths in enumerate(answer_lengths)]
        #combined = torch.cat(combined, 0)\
        results_packed, _ = pack_padded_sequence(results_tensor, answer_lengths, batch_first=True)

        outputs = self.linear(results_packed)

        return outputs #self.fc(combined)

    def _forward_free_cell(self, image_features, text_features, answer_lengths, states, skip=2):

        max_length = max(answer_lengths)
        output_tensor = Variable(torch.cuda.FloatTensor(len(answer_lengths),max_length,self.vocab_size))
        combined = torch.cat((image_features, text_features), 1)

        hiddens_tensor = Variable(torch.cuda.FloatTensor(len(answer_lengths),max_length,self.hidden_size))
        context_tensor = Variable(torch.cuda.FloatTensor(len(answer_lengths),max_length,self.hidden_size))
        hx, cx = states
        hx, cx = hx[0], cx[0]

        batch_size = image_features.size(0)
        if hx.size(0) != batch_size:
            hx = hx[:batch_size]
            cx = cx[:batch_size]

        inputs = combined
        for i in range(max_length):
            hx, cx = self.lstm( inputs, hx, cx)#, features_local)

            hiddens_tensor[ :, i, :] = hx
            context_tensor[ :, i, :] = cx

            # if i >= skip and (i % (skip + 1) == 0):
            #     hx = hx + hiddens_tensor[:, i - skip, :]
            #     cx = cx + context_tensor[:, i - skip, :]

            combined = torch.cat((hx, cx), 1)
            outputs = self.linear(combined)

            output_tensor[:,i,:] = outputs

            #tau = 1e-5
            #outputs = self.gumbel_sample(outputs, tau=tau)
            outputs = torch.max(outputs,1)[1]
            inputs = self.embed(outputs).view(outputs.size(0), -1)

            

        output_tensor, _ = pack_padded_sequence(output_tensor, answer_lengths, batch_first=True)
        return output_tensor

    def gumbel_sample(self,input, tau):
        noise = torch.rand(input.size()).cuda()
        noise.add_(1e-9).log_().neg_()
        noise.add_(1e-9).log_().neg_()
        noise = Variable(noise)
        x = (input + noise) / tau
        x = F.softmax(x.view(input.size(0), -1))
        return x.view_as(input)

    def beamSearch(self, features, states, n=1, diverse_gamma=0.0):
        features_global, features_local = features
        inputs = Variable(features_global.data, volatile=True)

        # onehot = torch.cuda.FloatTensor(1, self.vocab_size).fill_(0)
        # onehot[:,0] = 1
        # onehot = Variable(onehot, volatile=True)
        # inputs = self.embed(onehot)
        inputs = torch.cat((inputs, features_global), 1)

        hx, cx = states
        hx, cx = hx[0], cx[0]
        hx,cx = self.lstm_attention(inputs, hx,cx, features_local)     # (batch_size, 1, hidden_size)
        combined = torch.cat((hx,cx), 1)

        outputs = self.log_softmax(self.fc(combined))             # (batch_size, vocab_size)
        confidences, best_choices = outputs.topk(n)

        cached_states = [hx] * n
        cached_context = [cx] * n

        end_idx = self.vocab.word2idx["<end>"]
        
        best_choices = best_choices[0]
        terminated_choices = []
        terminated_confidences = []

        # one loop for each word
        for word_index in range(50):
            #print(best_choices)
            best_list = [None] * n * (n - len(terminated_choices))
            best_confidence = [None] * n * (n - len(terminated_choices))
            best_states = [None] * n * (n - len(terminated_choices))
            best_context = [None] * n * (n - len(terminated_choices))

            for choice_index, choice in enumerate(best_choices):

                input_choice = choice[word_index] if word_index != 0 else choice

                onehot = torch.cuda.FloatTensor(1, self.vocab_size).fill_(0)
                onehot[0][input_choice.data] = 1
                onehot = Variable(onehot, volatile=True)

                inputs = self.embed(onehot)
                inputs = torch.cat((inputs, features_global), 1)

                current_confidence = confidences[:,choice_index]
                out_states, out_context = self.lstm_attention(inputs,
                 cached_states[choice_index], cached_context[choice_index], features_local) # (batch_size, 1, hidden_size)

                combined = torch.cat((out_states,out_context), 1)
                outputs = self.log_softmax(self.fc(combined))                       # (batch_size, vocab_size)

                # pick n best nodes for each possible choice
                inner_confidences, inner_best_choices = outputs.topk(n)
                inner_confidences, inner_best_choices = inner_confidences.squeeze(), inner_best_choices.squeeze()
                for rank, inner_choice in enumerate(inner_best_choices):
                    if word_index != 0:
                        choice = best_choices[choice_index]
                    item = torch.cat((choice, inner_choice))
                    position = choice_index * n + rank
                    confidence = current_confidence + inner_confidences[rank] - (diverse_gamma * (rank + 1))

                    best_list[position]   = item
                    best_states[position] = out_states
                    best_context[position] = out_context
                    best_confidence[position] = current_confidence + inner_confidences[rank] - (diverse_gamma * (rank + 1))


            best_confidence_tensor = torch.cat(best_confidence, 0)
            _ , topk = best_confidence_tensor.topk(n - len(terminated_choices))

            topk_index = topk.data.int().cpu().numpy()

            ## Filter nodes that contains termination token '<end>'
            best_choices_index = [ index for index in topk_index if end_idx not in best_list[index].data.int() ]

            terminated_index   = list(set(topk_index) - set(best_choices_index))
            terminated_choices.extend([ best_list[index] for index in terminated_index ])
            terminated_confidences.extend([ best_confidence_tensor[index].data.cpu().numpy()[0] for index in terminated_index ])

            ## If there is still nodes to evaluate
            if len(best_choices_index) > 0:
                best_choices  = [ best_list[index] for index in best_choices_index ]
                cached_states = [ best_states[index] for index in best_choices_index ]
                cached_context = [ best_context[index] for index in best_choices_index ]
                confidences   = [ best_confidence_tensor[index] for index in best_choices_index ]
                confidences   = torch.cat(confidences,0).unsqueeze(0)
            else:
                break

        #print(best_choices)
        if len(best_choices_index) > 0:
            terminated_choices.extend([ best_list[index] for index in best_choices_index ])
            terminated_confidences.extend([ best_confidence_tensor[index].data.cpu().numpy()[0] for index in best_choices_index ])

        return terminated_choices,terminated_confidences
