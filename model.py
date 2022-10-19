from memory import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from DNC.dnc import *

class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                 dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
                                         hidden_size * self.num_directions)

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]  # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
        # (batch, hidden_size)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


control = 1
class EncoderLSTMUse(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                 dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTMUse, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size + control
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(self.hidden_size * self.num_directions,
                                         self.hidden_size * self.num_directions)

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]  # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
        # (batch, hidden_size)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax 
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size,
                 dropout_ratio, feature_size=2048):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, output_action_size)

    def forward(self, action, feature, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        action_embeds = self.embedding(action)  # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze()
        concat_input = torch.cat((action_embeds, feature), 1)  # (batch, embedding_size+feature_size)
        drop = self.drop(concat_input)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        logit = self.decoder2action(h_tilde)
        return h_1, c_1, alpha, logit





class DecoderHistN2N(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions.
        Has basic memory module attached which is updated and queried every step.'''

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size,
                 dropout_ratio, mem_embed_size=206, mem_output_size=206, mem_size=64, feature_size=2048):
        super(DecoderHistN2N, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.memory_module = FIFOMemoryN2N(feature_size + embedding_size, hidden_size, 20).cuda()
        self.B = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        #self.attn2 = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, output_action_size)
        #self.decoder2action = nn.Linear(2 * hidden_size, output_action_size)



    def forward(self, action, feature, h_0, c_0, ctx, memory=None, ctx_mask=None):
        #C
        action_embeds = self.embedding(action)  # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze()
        concat_input = torch.cat((action_embeds, feature), 1)  # (batch, embedding_size+feature_size)
        #history, new_memory = self.memory_module(concat_input.unsqueeze(1), concat_input.unsqueeze(1), memory)
        #concat_input = self.combine_history(concat_input, history.squeeze(1))
        drop = self.drop(concat_input)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        h_1_drop = self.drop(h_1)
        u = self.B(h_1_drop)
        output, new_memory = self.memory_module(concat_input.unsqueeze(1), u.unsqueeze(1), memory)
        o_drop = self.drop(output)
        v = o_drop + h_1_drop
        h_tilde, alpha = self.attention_layer(v, ctx, ctx_mask)
        logit = self.decoder2action(h_tilde)
        #logit = self.decoder2action(h_tilde.add(memory_output).mul(0.5))
        return h_1, c_1, new_memory, alpha, logit

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.memory_module.parameters():
            param.requires_grad = True
        self.decoder2action.weight.requires_grad = True
        self.B.weight.requires_grad = True
        for param in self.attention_layer.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class DecoderHistN2NAttn(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions.
        Has basic memory module attached which is updated and queried every step.'''

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size,
                 dropout_ratio, mem_embed_size=206, mem_output_size=206, mem_size=64, feature_size=2048):
        super(DecoderHistN2NAttn, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.memory_module = FIFOMemoryN2N(feature_size + embedding_size, hidden_size, 20).cuda()
        self.B = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.attn2 = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(2 * hidden_size, output_action_size)



    def forward(self, action, feature, h_0, c_0, ctx, memory=None, ctx_mask=None):
        #C_attn
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''

        action_embeds = self.embedding(action)  # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze()
        concat_input = torch.cat((action_embeds, feature), 1)  # (batch, embedding_size+feature_size)
        #history, new_memory = self.memory_module(concat_input.unsqueeze(1), concat_input.unsqueeze(1), memory)
        #concat_input = self.combine_history(concat_input, history.squeeze(1))
        drop = self.drop(concat_input)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        h_1_drop = self.drop(h_1)
        q = self.B(h_1_drop)
        output, new_memory = self.memory_module(concat_input.unsqueeze(1), q.unsqueeze(1), memory)
        o_drop = self.drop(output)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        o_tilde, _ = self.attn2(o_drop, ctx, ctx_mask)
        logit = self.decoder2action(torch.cat((h_tilde, o_tilde), 1))
        return h_1, c_1, new_memory, alpha, logit



    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.memory_module.parameters():
            param.requires_grad = True
        self.decoder2action.weight.requires_grad = True
        self.B.weight.requires_grad = True
        for param in self.attention_layer.parameters():
            param.requires_grad = True
        for param in self.attn2.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DecoderSingleDNC(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions.
        Has basic memory module attached which is updated and queried every step.'''

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size,
                 dropout_ratio, mem_embed_size=1500, rh=1, mem_size=20, feature_size=2048):
        super(DecoderSingleDNC, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.dnc = DNC(input_size=embedding_size + feature_size,
                        output_size= hidden_size,
                        hidden_size=hidden_size,
                        rnn_type='lstm',
                        num_layers=1,
                        num_hidden_layers=1,
                        nr_cells=mem_size,
                        cell_size=mem_embed_size,
                        read_heads=rh,
                        batch_first=True,
                        gpu_id=0,
                        dropout=0.25)
        self.attention_layer = SoftDotAttention(self.hidden_size)
        self.decoder2action = nn.Linear(self.hidden_size, output_action_size)



    def forward(self, action, feature, chx_0, ctx, memory, read_vectors, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        action_embeds = self.embedding(action)  # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze()
        concat_input = torch.cat((action_embeds, feature), 1)  # (batch, embedding_size+feature_size)
        drop = self.drop(concat_input)
        output, (chx_1, memory, read_vectors) = self.dnc(drop, (chx_0, memory, read_vectors))

        output_drop = self.drop(output)
        h_tilde, alpha = self.attention_layer(output_drop, ctx, ctx_mask)
        logit = self.decoder2action(h_tilde)
        return chx_1, memory, read_vectors, alpha, logit

    def freeze(self):
        #self.dnc.freeze()
        self.embedding.weight.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True