import json
import os
import sys
import numpy as np
import random
import time
import gc
import copy
import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from agent import Seq2SeqAgent

from env import R2RBatch
from utils import padding_idx

class BaseMemoryAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.decoder = None
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)
    def get_obs(self):
        raise NotImplementedError

    def rollout(self, mem, loss, obs):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, persist, batch_size):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print('Testing %s' % self.__class__.__name__)
        looped = False
        zero = self.decoder.memory_module.initState(batch_size).cuda()
        while True:
            obs = self.get_obs()
            if persist:
                (_, mem) = self.rollout(mem=zero, loss=False, obs=obs)
                self.env.set_sim()
            else:
                mem = zero
            (trajs, _) = self.rollout(mem=mem, loss=True, obs=obs)
            for traj in trajs:
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break




class Seq2SeqMemoryAgent(BaseMemoryAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
      (0,-1, 0), # left
      (0, 1, 0), # right
      (0, 0, 1), # up
      (0, 0,-1), # down
      (1, 0, 0), # forward
      (0, 0, 0), # <end>
      (0, 0, 0), # <start>
      (0, 0, 0)  # <ignore>
    ]
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, episode_len=20):
        super(Seq2SeqMemoryAgent, self).__init__(env, results_path)
        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.model_actions.index('<ignore>'))

    @staticmethod
    def n_inputs():
        return len(Seq2SeqMemoryAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqMemoryAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            ix,heading_chg,elevation_chg = ob['teacher']
            if heading_chg > 0:
                a[i] = self.model_actions.index('right')
            elif heading_chg < 0:
                a[i] = self.model_actions.index('left')
            elif elevation_chg > 0:
                a[i] = self.model_actions.index('up')
            elif elevation_chg < 0:
                a[i] = self.model_actions.index('down')
            elif ix > 0:
                a[i] = self.model_actions.index('forward')
            elif ended[i]:
                a[i] = self.model_actions.index('<ignore>')
            else:
                a[i] = self.model_actions.index('<end>')
        return Variable(a, requires_grad=False).cuda()

    def get_obs(self):
        return np.array(self.env.reset())

    def rollout(self, mem, loss=True, obs=None):
        batch_size = len(obs)
        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]
        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        # Initial action
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'),
                    requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        env_action = [None] * batch_size
        for t in range(self.episode_len):
            f_t = self._feature_variable(perm_obs) # Image features from obs
            h_t,c_t, mem, alpha,logit = self.decoder(a_t.view(-1, 1), f_t, h_t, c_t, ctx, mem, seq_mask)
            # Mask outputs where agent can't move forward
            for i,ob in enumerate(perm_obs):
                if len(ob['navigableLocations']) <= 1:
                    logit[i, self.model_actions.index('forward')] = -float('inf')

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            if loss:
                self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()            # sampling an action from model
            else:
                sys.exit('Invalid feedback option')

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            # Early exit if all ended
            if ended.all():
                break
        if loss:
            self.losses.append(self.loss.item() / self.episode_len)
        return (traj, mem)


    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, persist=False, batch_size=100):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(Seq2SeqMemoryAgent, self).test( persist=persist, batch_size=batch_size)

    def eval(self, batch_size, j):
        self.encoder.eval()
        self.decoder.eval()
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        looped = False
        while True:
            mem = self.decoder.memory_module.initState(batch_size).cuda()
            obs = self.get_obs()
            for i in range(j):
                (_, mem) = self.rollout(mem=mem, loss=False, obs=obs)
                self.env.set_sim()
            (trajs, _) = self.rollout(mem=mem, loss=True, obs=obs)
            for traj in trajs:
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, feedback='teacher', persist=False, batch_size=100):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        self.train_s(encoder_optimizer, decoder_optimizer, n_iters, persist, batch_size=batch_size)
        return


    def train_s(self, encoder_optimizer, decoder_optimizer, n_iters, persist, batch_size):
        zero = self.decoder.memory_module.initState(batch_size).cuda()
        for iter in range(1, n_iters + 1):
            obs = self.get_obs()
            if persist:
                (_, mem) = self.rollout(mem=zero, loss=False, obs=obs)
                self.env.set_sim()
            else:
                mem = zero
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            (_, _) = self.rollout(mem, True, obs)
            self.loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            #mem = mem.detach()
            #self.decoder.memory_module.usage = self.decoder.memory_module.usage.detach()
        return

    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))





class BaseDNCAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self, memory, read_vectors, loss, obs):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def get_obs(self):
        raise NotImplementedError

    def test(self, persist):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print('Testing %s' % self.__class__.__name__)
        looped = False
        i = 0
        while True:
            obs = self.get_obs()
            if persist:
                (_, memory, read_vectors) = self.rollout(memory=None, read_vectors=None, loss=False, obs=obs)
                self.env.set_sim()
            else:
                memory = None
                read_vectors = None
            trajs, _, _ = self.rollout(memory, read_vectors, loss=True, obs=obs)
            for traj in trajs:
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break

    def eval(self, j):
        raise NotImplementedError

class Seq2SeqDNCAgent(BaseDNCAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
      (0,-1, 0), # left
      (0, 1, 0), # right
      (0, 0, 1), # up
      (0, 0,-1), # down
      (1, 0, 0), # forward
      (0, 0, 0), # <end>
      (0, 0, 0), # <start>
      (0, 0, 0)  # <ignore>
    ]
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, episode_len=20):
        super(Seq2SeqDNCAgent, self).__init__(env, results_path)
        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.model_actions.index('<ignore>'))

    @staticmethod
    def n_inputs():
        return len(Seq2SeqMemoryAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqMemoryAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            ix,heading_chg,elevation_chg = ob['teacher']
            if heading_chg > 0:
                a[i] = self.model_actions.index('right')
            elif heading_chg < 0:
                a[i] = self.model_actions.index('left')
            elif elevation_chg > 0:
                a[i] = self.model_actions.index('up')
            elif elevation_chg < 0:
                a[i] = self.model_actions.index('down')
            elif ix > 0:
                a[i] = self.model_actions.index('forward')
            elif ended[i]:
                a[i] = self.model_actions.index('<ignore>')
            else:
                a[i] = self.model_actions.index('<end>')
        return Variable(a, requires_grad=False).cuda()

    def get_obs(self):
        return np.array(self.env.reset())

    def rollout(self, memory, read_vectors, loss, obs):
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)
        h_t = h_t.unsqueeze(0)
        c_t = c_t.unsqueeze(0)
        chx = [(h_t, c_t) for x in range(self.decoder.dnc.num_layers)]

        # Initial action
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'),
                    requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env
        # Do a sequence rollout and calculate the loss
        self.loss = 0
        env_action = [None] * batch_size
        for t in range(self.episode_len):

            f_t = self._feature_variable(perm_obs) # Image features from obs
            chx, memory, read_vectors, alpha, logit = self.decoder(
                a_t.view(-1, 1), f_t, chx, ctx, memory, read_vectors, seq_mask)
            # Mask outputs where agent can't move forward
            for i,ob in enumerate(perm_obs):
                if len(ob['navigableLocations']) <= 1:
                    logit[i, self.model_actions.index('forward')] = -float('inf')

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            if loss:
                self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()            # sampling an action from model
            else:
                sys.exit('Invalid feedback option')

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            if loss:
                for i,ob in enumerate(perm_obs):
                    if not ended[i]:
                        traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            # Early exit if all ended
            if ended.all():
                break
        if loss:
            self.losses.append(self.loss.item() / self.episode_len)
        return traj, memory, read_vectors

    def test(self, use_dropout=False, persist=False, feedback='argmax', allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(Seq2SeqDNCAgent, self).test(persist)

    def eval(self, j):
        self.encoder.eval()
        self.decoder.eval()
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        looped = False
        while True:
            obs = self.get_obs()
            mem = None
            rv = None
            for i in range(j):
                (_, mem, rv) = self.rollout(memory=mem, read_vectors=rv, loss=False, obs=obs)
                self.env.set_sim()
            (trajs, _, _) = self.rollout(memory=mem, read_vectors=rv, loss=True, obs=obs)
            for traj in trajs:
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, persist=False, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        for iter in range(1, n_iters + 1):
            obs = self.get_obs()
            if persist:
                (_, mem, rv) = self.rollout(memory=None, read_vectors=None, loss=False, obs=obs)
                self.env.set_sim()
            else:
                mem = None
                rv = None
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            _, _, _ = self.rollout(memory=mem, read_vectors=rv, loss=True, obs=obs)
            self.loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
        return




    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))


