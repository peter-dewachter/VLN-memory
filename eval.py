''' Evaluation of agent trajectories '''

import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
import pandas as pd

pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from utils import load_datasets, load_nav_graphs, read_vocab, Tokenizer
from agent import BaseAgent, StopAgent, RandomAgent, ShortestAgent
from model import EncoderLSTM, DecoderHistN2N, DecoderHistN2NAttn, DecoderSingleDNC
from my_agent import *


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits):
        self.error_margin = 3.0
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        for item in load_datasets(splits):
            self.gt[item['path_id']] = item
            self.scans.append(item['scan'])
            self.instr_ids += ['%d_%d' % (item['path_id'], i) for i in range(3)]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule). '''
        gt = self.gt[int(instr_id.split('_')[0])]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            if prev[0] != curr[0]:
                try:
                    self.graphs[gt['scan']][prev[0]][curr[0]]
                except KeyError as err:
                    print('Error: The provided trajectory moves from %s to %s but the navigation graph contains no ' \
                          'edge between these viewpoints. Please ensure the provided navigation trajectories ' \
                          'are valid, so that trajectory length can be accurately calculated.' % (prev[0], curr[0]))
                    raise
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_path_lengths'].append(self.distances[gt['scan']][start][goal])

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id'])
                    self._score_item(item['instr_id'], item['trajectory'])
        assert len(instr_ids) == 0, 'Trajectories not provided for %d instruction ids: %s' % (len(instr_ids), instr_ids)
        assert len(self.scores['nav_errors']) == len(self.instr_ids)
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])

        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])

        spls = []
        for err, length, sp in zip(self.scores['nav_errors'], self.scores['trajectory_lengths'],
                                   self.scores['shortest_path_lengths']):
            if err < self.error_margin:
                spls.append(sp / max(length, sp))
            else:
                spls.append(0)

        score_summary = {
            'length': np.average(self.scores['trajectory_lengths']),
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle success_rate': float(oracle_successes) / float(len(self.scores['oracle_errors'])),
            'success_rate': float(num_successes) / float(len(self.scores['nav_errors'])),
            'spl': np.average(spls)
        }

        assert score_summary['spl'] <= score_summary['success_rate']
        return score_summary, self.scores


RESULT_DIR = 'tasks/R2R/results/'


def eval_simple_agents():
    ''' Run simple baselines on each split. '''
    for split in ['train', 'val_seen', 'val_unseen']:
        env = R2RBatch(None, batch_size=1, splits=[split])
        ev = Evaluation([split])

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score(outfile)
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)

import my_train as train
DIR = 'C_persist/'
LOAD_DIR = 'tasks/R2R/snapshots/'
EVAL_DIR = 'tasks/R2R/eval/'


def test(dir,it,bs):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)
    vocab = read_vocab(train.TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=train.MAX_INPUT_LENGTH)
    # load agent
    agent = load_mem_agent(vocab, dir, it, 0)
    agent.feedback = "argmax"
    # env
    envs = [R2RBatch(train.features, batch_size=bs, splits=[split], tokenizer=tok)
            for split in ['val_seen', 'val_unseen']]
    evals = [Evaluation([split]) for split in ['val_seen', 'val_unseen']]

    data_log = defaultdict(list)

    for i, split in enumerate(['val_seen', 'val_unseen']):
        agent.env = envs[i]
        # Get validation distance from goal under test evaluation conditions
        agent.test(batch_size=bs, persist=False)
        agent.results_path = EVAL_DIR + dir + split + "_it" + '.json'
        agent.write_results()
        score_summary, _ = evals[i].score(agent.results_path)
        for metric, val in score_summary.items():
            data_log['%s %s' % (split, metric)].append(val)

    df = pd.DataFrame(data_log)
    df_path = '%s%s_%s_log.csv' % (EVAL_DIR, dir[:-1], str(it))
    df.to_csv(df_path)

def eval_my_agent(dir, it, bs, n, name=''):
    # setup
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)
    vocab = read_vocab(train.TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=train.MAX_INPUT_LENGTH)
    # load agent
    agent = load_mem_agent(vocab, dir, it, n)
    agent.feedback = "argmax"
    #env
    envs = [R2RBatch(train.features, batch_size=bs, splits=[split], tokenizer=tok)
            for split in ['val_seen', 'val_unseen']]
    evals = [Evaluation([split]) for split in ['val_seen', 'val_unseen']]

    data_log = defaultdict(list)

    for i, split in enumerate(['val_seen', 'val_unseen']):
        agent.env = envs[i]
        for j in range(3):
            print(j)
            # Get validation distance from goal under test evaluation conditions
            agent.eval(batch_size=bs, j=j)
            agent.results_path = EVAL_DIR + dir + split + "_it" + str(j) + '.json'
            agent.write_results()
            score_summary, _ = evals[i].score(agent.results_path)
            for metric, val in score_summary.items():
                data_log['%s %s' % (split, metric)].append(val)

    df = pd.DataFrame(data_log)
    df_path = '%s%s_%s%s_log.csv' % (EVAL_DIR, dir[:-1], name, str(it))
    df.to_csv(df_path)

def eval_dnc_agent(dir, it, bs):
    # setup
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)
    vocab = read_vocab(train.TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=train.MAX_INPUT_LENGTH)
    # load agent
    agent = load_mem_agent(vocab, dir, it, 3)
    agent.feedback = "argmax"
    #env
    envs = [R2RBatch(train.features, batch_size=bs, splits=[split], tokenizer=tok)
            for split in ['val_seen', 'val_unseen']]
    evals = [Evaluation([split]) for split in ['val_seen', 'val_unseen']]

    data_log = defaultdict(list)

    for i, split in enumerate(['val_seen', 'val_unseen']):
        agent.env = envs[i]
        for j in range(3):
            print(j)
            # Get validation distance from goal under test evaluation conditions
            agent.eval(j=j)
            agent.results_path = EVAL_DIR + dir + split + "_it" + str(j) + '.json'
            agent.write_results()
            score_summary, _ = evals[i].score(agent.results_path)
            for metric, val in score_summary.items():
                data_log['%s %s' % (split, metric)].append(val)

    df = pd.DataFrame(data_log)
    df_path = '%s%s_%s_log.csv' % (EVAL_DIR, dir[:-1], str(it))
    df.to_csv(df_path)



def load_mem_agent(vocab, dir, it, n):
    enc_hidden_size = train.hidden_size // 2 if train.bidirectional else train.hidden_size
    encoder = EncoderLSTM(len(vocab), train.word_embedding_size, enc_hidden_size, train.padding_idx,
                          train.dropout_ratio, bidirectional=train.bidirectional).cuda()
    if n == 0:
        decoder = DecoderHistN2N(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                                   train.action_embedding_size, train.hidden_size, train.dropout_ratio).cuda()
    elif n == 1:
        decoder = DecoderHistN2NAttn(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                                 train.action_embedding_size, train.hidden_size, train.dropout_ratio).cuda()
    else:
        decoder = DecoderSingleDNC(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                                 train.action_embedding_size, train.hidden_size, train.dropout_ratio).cuda()
    if n == 0 or n == 1:
        agent = Seq2SeqMemoryAgent(None, "myeval", encoder, decoder, train.max_episode_len)
    else:
        agent = Seq2SeqDNCAgent(None, "myeval", encoder, decoder, train.max_episode_len)

    agent.load(LOAD_DIR + dir + 'seq2seq_sample_train_enc_iter_' + str(it),
               LOAD_DIR + dir + 'seq2seq_sample_train_dec_iter_' + str(it))

    if not os.path.exists(EVAL_DIR + dir):
        os.makedirs(EVAL_DIR + dir)
    return agent





if __name__ == '__main__':
    #eval_simple_agents()
    # eval_seq2seq()
    eval_my_agent('C2/', 19800, 100, 0, 'c2')
    eval_my_agent('C_attn/', 19800, 100, 1, 'c_attnnn')

    #test('C2/', 19800, 100)
    #eval_seq2seq()
    #eval_my_agent('C2_persist/', 14600, 100, 0)
    #eval_my_agent('C2_persist/', 4500, 100, 0)
    #eval_my_agent('C2_persist/', 18800, 100, 0)

    #eval_my_agent('C_attn/', 15200, 100, 1)
    #eval_my_agent('C_attn_persist/', 17200, 100, 1)
    #eval_my_agent('C_attn_persist/', 17300, 100, 1)
    #eval_my_agent('C_attn_persist/', 17900, 100, 1)




    #eval_dnc_agent('dnc/', 14700, 100)
    #eval_dnc_agent('dnc/', 11400, 100)
    #eval_dnc_agent('dnc/', 10500, 100)

    #eval_dnc_agent('dnc_persist/', 13100, 100)
    #eval_dnc_agent('dnc_persist/', 2700, 100)
    #eval_dnc_agent('dnc_persist2/', 10400, 100)
    #eval_dnc_agent('dnc_persist2/', 11600, 100)






