import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from encoding import *


class ParallelMemory(nn.Module):

    def __init__(self, input_size, query_size, mem_embed_size, output_size, memory_size):
        super(ParallelMemory, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.mem_embed_size = mem_embed_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.internal_size = 64
        # create parameters according to different type of weight tying
        self.I = nn.Linear(self.input_size, self.mem_embed_size)
        self.W1 = nn.Linear(self.input_size, self.internal_size)
        self.W2 = nn.Linear(self.mem_embed_size, self.internal_size)

        self.F1 = nn.Linear(self.query_size, self.internal_size)
        self.F2 = nn.Linear(self.mem_embed_size, self.internal_size)

        self.R1 = nn.Linear(self.query_size, self.internal_size)
        self.R2 = nn.Linear(self.mem_embed_size, self.internal_size)
        self.O = nn.Linear(self.mem_embed_size, self.output_size)

        self.read_only = False
        #self.memory = Variable(torch.zeros(self.memory_size, self.mem_embed_size), requires_grad=False)

    def forward(self, input, query, memory):
        output = self.read(query, memory)
        if not self.read_only:
            new_memory = self.write2(input, memory)
            return output, new_memory
        return output, memory

    def initState(self, batch_size):
        mem = torch.zeros(batch_size, self.memory_size, self.mem_embed_size)
        return I.xavier_uniform_(mem)

    def forget(self, query, memory):
        Fq = self.F1(query)
        Fm = self.F2(memory)
        mul = torch.bmm(Fq, Fm.transpose(2, 1))
        f = torch.sigmoid(mul)
        mem = torch.mul(f.transpose(1,2), memory)
        return mem

    def write(self, input, memory):
        Wi = self.W1(input)
        Wm = self.W2(memory)
        mul = torch.bmm(Wi, Wm.transpose(1, 2))
        p = F.softmax(mul, 2)
        embedded_input = self.I(input)
        i = torch.mul(p.transpose(1, 2), embedded_input)
        result_memory = memory + i
        return result_memory



    def read(self, query, memory):
        Rq = self.R1(query)
        Rm = self.R2(memory)
        mul = torch.bmm(Rq, Rm.transpose(2, 1))
        w = F.softmax(mul, 2)
        i = torch.mul(w.transpose(1, 2), memory)
        out = torch.sum(i, dim=1)
        return self.O(out)



class FIFOMemory(nn.Module):
    def __init__(self, mem_embed_size, memory_size):
        super(FIFOMemory, self).__init__()
        self.memory_size = memory_size
        self.mem_embed_size = mem_embed_size
        self.cossim = torch.nn.CosineSimilarity(dim=2, eps=1e-08)

        self.index = 0

        # create parameters according to different type of weight tyi

        self.read_only = False
        #self.memory = Variable(torch.zeros(self.memory_size, self.mem_embed_size), requires_grad=False)

    def forward(self, input, query, memory):
        output = self.read(query, memory)
        if not self.read_only:
            new_memory = self.write2(input, memory)
            return output, new_memory
        return output, memory

    def initState(self, batch_size):
        mem = torch.zeros(batch_size, self.memory_size, self.mem_embed_size)
        return mem

    def forget(self, query, memory):
        return

    def write(self, input, memory):
        #print(input.shape)
        #print(memory.shape)
        memory[:, self.index, :] = input
        new_memory = memory
        self.index = (self.index + 1) % self.memory_size
        return new_memory

    def write2(self, input, memory):
        new_memory = torch.cat((input, memory), dim=1)[:,0:self.memory_size]
        return new_memory

    def read(self, query, memory):
        sim = F.cosine_similarity(query, memory, dim=2)
        w = F.softmax(sim, 1).unsqueeze(1)

        i = torch.mul(w.transpose(1, 2), memory)
        out = torch.sum(i, dim=1)
        return out

    def hasIndex(self):
        return True

class FIFOMemoryAttnRead(nn.Module):
    def __init__(self, mem_embed_size, query_size, memory_size):
        super(FIFOMemoryAttnRead, self).__init__()
        self.memory_size = memory_size
        self.mem_embed_size = mem_embed_size
        self.cossim = torch.nn.CosineSimilarity(dim=2, eps=1e-08)
        self.query_size = query_size
        self.index = 0
        self.internal_size = 512
        self.R1 = nn.Linear(query_size, self.internal_size)
        self.R2 = nn.Linear(mem_embed_size, self.internal_size)

        # create parameters according to different type of weight tyi

        self.read_only = False
        #self.memory = Variable(torch.zeros(self.memory_size, self.mem_embed_size), requires_grad=False)

    def forward(self, input, query, memory):
        output = self.read(query, memory)
        if not self.read_only:
            new_memory = self.write2(input, memory)
            return output, new_memory
        return output, memory

    def initState(self, batch_size):
        mem = torch.zeros(batch_size, self.memory_size, self.mem_embed_size)
        return mem

    def forget(self, query, memory):
        return

    def write2(self, input, memory):
        new_memory = torch.cat((input, memory), dim=1)[:,0:self.memory_size]
        return new_memory

    def read(self, query, memory):
        Rq = self.R1(query)
        Rm = self.R2(memory)
        mul = torch.bmm(Rq, Rm.transpose(2, 1))
        w = F.softmax(mul, 2)
        i = torch.mul(w.transpose(1, 2), memory)
        out = torch.sum(i, dim=1)
        return out

    def hasIndex(self):
        return True

class FIFOMemoryN2N(nn.Module):
    def __init__(self, mem_embed_size, query_size, memory_size):
        super(FIFOMemoryN2N, self).__init__()
        self.memory_size = memory_size
        self.mem_embed_size = mem_embed_size
        self.query_size = query_size
        self.index = 0
        self.pos_encoding = PositionalEncoding1D(self.query_size)
        self.A = nn.Linear(mem_embed_size, self.query_size)
        self.C = nn.Linear(mem_embed_size, self.query_size)

        # create parameters according to different type of weight tyi

        self.read_only = False
        #self.memory = Variable(torch.zeros(self.memory_size, self.mem_embed_size), requires_grad=False)


    def forward(self, input, query, memory):
        output = self.read(query, memory)
        if not self.read_only:
            new_memory = self.write2(input, memory)
            return output, new_memory

        return output, memory

    def initState(self, batch_size):
        mem = torch.zeros(batch_size, self.memory_size, self.mem_embed_size)
        return mem

    def forget(self, query, memory):
        return

    def write2(self, input, memory):
        new_memory = torch.cat((input, memory), dim=1)[:,0:self.memory_size]
        return new_memory

    def read(self, query, memory):
        Am = self.A(memory)
        Am = Am + self.pos_encoding(Am)
        mul = torch.bmm(query, Am.transpose(2, 1))
        w = F.softmax(mul, 2)
        Cm = self.C(memory)
        Cm = Cm + self.pos_encoding(Cm)
        i = torch.mul(w.transpose(1, 2), Cm)
        out = torch.sum(i, dim=1)
        return out

    def hasIndex(self):
        return True

class KVMemory(nn.Module):
    def __init__(self, input_size, query_size, key_embed_size, val_embed_size, output_size, memory_size):
        super(KVMemory, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.key_embed_size = key_embed_size
        self.val_embed_size = val_embed_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.internal_size = 64
        # create parameters according to different type of weight tying
        self.KI = nn.Linear(self.input_size, self.key_embed_size)
        self.VI = nn.Linear(self.input_size, self.val_embed_size)
        self.W1 = nn.Linear(self.input_size, self.internal_size)
        self.W2 = nn.Linear(self.key_embed_size, self.internal_size)
        self.V = nn.Linear(self.input_size, self.val_embed_size)

        self.F1 = nn.Linear(self.query_size, self.internal_size)
        self.F2 = nn.Linear(self.key_embed_size, self.internal_size)

        self.R1 = nn.Linear(self.query_size, self.internal_size)
        self.R2 = nn.Linear(self.key_embed_size, self.internal_size)
        self.O = nn.Linear(self.val_embed_size, self.output_size)

        self.read_only = False
        #self.memory = Variable(torch.zeros(self.memory_size, self.mem_embed_size), requires_grad=False)

    def forward(self, input, query, memory):
        if not self.read_only:
            #memory = self.forget(query, memory)
            memory = self.write(input, memory)
        output = self.read(query, memory)
        return output, memory

    def initMemory(self, batch_size):
        return (
            torch.zeros(batch_size, self.memory_size, self.key_embed_size),
            torch.zeros(batch_size, self.memory_size, self.val_embed_size)
        )

    def forget(self, query, mem):
        keys, vals = mem
        Fq = self.F1(query)
        Fm = self.F2(keys)
        mul = torch.bmm(Fq, Fm.transpose(2, 1))
        w = torch.sigmoid(mul)
        new_keys = torch.mul(w.transpose(1,2), keys)
        new_vals = torch.mul(w.transpose(1,2), vals)
        return new_keys, new_vals

    def write(self, input, mem):
        keys, vals = mem
        Wi = self.W1(input)
        Wk = self.W2(keys)
        mul = torch.bmm(Wi, Wk.transpose(1, 2))
        w = F.softmax(mul, 2)
        k_embedded_input = self.KI(input)
        in_k = torch.mul(w.transpose(1, 2), k_embedded_input)
        new_keys = keys + in_k
        v_embedded_input = self.V(input)
        in_v = torch.mul(w.transpose(1,2), v_embedded_input)
        new_vals = vals + in_v
        return new_keys, new_vals

    def read(self, query, mem):
        keys, vals = mem
        Rq = self.R1(query)
        Rm = self.R2(keys)
        mul = torch.bmm(Rq, Rm.transpose(2, 1))
        p = F.softmax(mul, 2)
        i = torch.mul(p.transpose(1, 2), vals)
        out = torch.sum(i, dim=1)
        return self.O(out)




class UsageMemory(nn.Module):

    def __init__(self, input_size, query_size, mem_embed_size, output_size, memory_size):
        super(UsageMemory, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.mem_embed_size = mem_embed_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.internal_size = 64
        # create parameters according to different type of weight tying
        self.I = nn.Linear(self.input_size, self.mem_embed_size)
        self.W1 = nn.Linear(self.mem_embed_size, self.internal_size)
        self.W2 = nn.Linear(self.mem_embed_size, self.internal_size)
        self.P = nn.Linear(self.memory_size, self.memory_size)

        self.F1 = nn.Linear(self.query_size, self.internal_size)
        self.F2 = nn.Linear(self.mem_embed_size, self.internal_size)

        self.R1 = nn.Linear(self.query_size, self.internal_size)
        self.R2 = nn.Linear(self.mem_embed_size, self.internal_size)
        self.O = nn.Linear(self.mem_embed_size, self.output_size)

        self.read_only = False

    def forward(self, state_vector, free_gates, input, query, memory):
        read_weights, write_weights, usage = state_vector
        if not self.read_only:
            # memory = self.forget(query, memory)
            new_usage, new_write_weights, memory = self.write(read_weights, write_weights, usage, free_gates, input, memory)
        new_read_weights, output = self.read(query, memory)

        if not self.read_only:
            return (new_read_weights, new_write_weights, new_usage), output, memory
        else:
            return state_vector, output, memory

    def initState(self, batch_size):
        read_weights = torch.zeros(batch_size, self.memory_size)
        write_weights = torch.zeros(batch_size, self.memory_size)
        usage = torch.zeros(batch_size, self.memory_size)
        memory = torch.zeros(batch_size, self.memory_size, self.mem_embed_size)
        return (read_weights, write_weights, usage), memory

    def forget(self, query, memory):
        Fq = self.F1(query)
        Fm = self.F2(memory)
        mul = torch.bmm(Fq, Fm.transpose(2, 1))
        f = torch.sigmoid(mul)
        mem = torch.mul(f.transpose(1, 2), memory)
        return mem

    def write(self, read_weights, write_weights, usage, free_gates, input, memory):
        embedded_input = self.I(input)
        Wi = self.W1(embedded_input)
        Wm = self.W2(memory)
        mul = torch.bmm(Wi, Wm.transpose(1, 2))
        w_content = F.softmax(mul, 2)
        usage, w_alloc = self.allocation_weight(read_weights, write_weights, usage, free_gates)

        w = (w_content + w_alloc.unsqueeze(1)) * 0.5
        i = torch.mul(w.transpose(1, 2), embedded_input)
        result_memory = memory + i
        return usage, w.squeeze(1), result_memory


    def allocation_weight(self, read_weights, write_weights, usage, free_gates):
        usage = 10e-6 + (1 - 10e-6) * usage
        psi = 1 - free_gates * read_weights
        usage = usage + write_weights - usage * write_weights
        usage = usage * psi
        sorted_usage, phi = torch.topk(usage, self.memory_size, dim=1, largest=False)
        sorted_allocation_weights = (1 - sorted_usage) * torch.cumprod(sorted_usage, dim=1)
        # construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
        _, phi_rev = torch.topk(phi, k=self.memory_size, dim=1, largest=False)
        allocation_weights = sorted_allocation_weights.gather(1, phi_rev.long())
        return usage, allocation_weights



    def read(self, query, memory):
        Rq = self.R1(query)
        Rm = self.R2(memory)
        mul = torch.bmm(Rq, Rm.transpose(2, 1))
        w = F.softmax(mul, 2)
        i = torch.mul(w.transpose(1, 2), memory)
        out = torch.sum(i, dim=1)
        return w.squeeze(1), self.O(out)