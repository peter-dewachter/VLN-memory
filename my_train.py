import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import tracemalloc

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import gc

from utils import read_vocab, write_vocab, build_vocab, Tokenizer, padding_idx, timeSince
from env import R2RBatch
from model import EncoderLSTM, AttnDecoderLSTM, EncoderLSTMUse, DecoderSingleDNC, DecoderHistN2N, DecoderHistN2NAttn
from my_agent import Seq2SeqMemoryAgent, Seq2SeqDNCAgent
from agent import Seq2SeqAgent
from eval import Evaluation

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'
DIR = 'C_attention/'

RESULT_DIR = 'tasks/R2R/results/'
SNAPSHOT_DIR = 'tasks/R2R/snapshots/'
PLOT_DIR = 'tasks/R2R/plots/'


IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80

features = IMAGENET_FEATURES
batch_size = 100
max_episode_len = 20
word_embedding_size = 256
action_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
feedback_method = 'sample'  # teacher or sample
learning_rate = 0.0001
weight_decay = 0.0005
n_iters = 5000 if feedback_method == 'teacher' else 20000
model_prefix = 'seq2seq_%s' % (feedback_method)


def train_mem(train_env, encoder, decoder, n_iters, dir, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''

    agent = Seq2SeqMemoryAgent(train_env, "", encoder, decoder, max_episode_len)
    print('Training with %s feedback' % feedback_method)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    data_log = defaultdict(list)
    start = time.time()


    #mem_state = (mem_state[0].cuda(), mem_state[1].cuda(), mem_state[2].cuda())
    for idx in range(0, n_iters, log_every):
        decoder.memory_module.read_only = False
        interval = min(log_every, n_iters - idx)
        iter = idx + interval
        data_log['iteration'].append(iter)
        # Train for log_every interval
        agent.train(encoder_optimizer, decoder_optimizer, interval,
                          batch_size=batch_size, persist=False, feedback=feedback_method)
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        #decoder.memory_module.read_only = True
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR + dir, model_prefix, env_name, iter)
            # Get validation loss under the same conditions as training
            #agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True, mem_state=mem_state, memory=mem)
            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True,
                       batch_size=batch_size)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            # Get validation distance from goal under test evaluation conditions
            #agent.test(use_dropout=False, feedback='argmax', mem_state=mem_state, memory=mem)
            agent.test(use_dropout=True, feedback='argmax', allow_cheat=True,
                       batch_size=batch_size)
            agent.write_results()
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric, val in score_summary.items():
                data_log['%s %s' % (env_name, metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)
        agent.env = train_env

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                   iter, float(iter) / n_iters * 100, loss_str))

        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s_log.csv' % (PLOT_DIR + dir, model_prefix)
        df.to_csv(df_path)
        sd = SNAPSHOT_DIR + dir
        split_string = "-".join(train_env.splits)
        enc_path = '%s%s_%s_enc_iter_%d' % (sd, model_prefix, split_string, iter)
        dec_path = '%s%s_%s_dec_iter_%d' % (sd, model_prefix, split_string, iter)
        agent.save(enc_path, dec_path)


def train_persmem(train_env, encoder, decoder, n_iters, tgtdir, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''

    agent = Seq2SeqMemoryAgent(train_env, "", encoder, decoder, max_episode_len)
    print('Training with %s feedback' % feedback_method)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    data_log = defaultdict(list)
    start = time.time()

    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)
        iter = idx + interval
        data_log['iteration'].append(iter)
        # Train for log_every interval
        agent.feedback = feedback_method
        agent.train(encoder_optimizer, decoder_optimizer, interval, feedback=feedback_method, persist=True,
                    batch_size=batch_size)
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR + tgtdir, model_prefix, env_name, iter)
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True,
                       batch_size=batch_size, persist=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=True, feedback='argmax', allow_cheat=True,
                       batch_size=batch_size, persist=True)
            agent.write_results()
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric, val in score_summary.items():
                data_log['%s %s' % (env_name, metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)
        agent.env = train_env

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                   iter, float(iter) / n_iters * 100, loss_str))

        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s_log.csv' % (PLOT_DIR + tgtdir, model_prefix)
        df.to_csv(df_path)
        sd = SNAPSHOT_DIR + tgtdir
        split_string = "-".join(train_env.splits)
        enc_path = '%s%s_%s_enc_iter_%d' % (sd, model_prefix, split_string, iter)
        dec_path = '%s%s_%s_dec_iter_%d' % (sd, model_prefix, split_string, iter)
        agent.save(enc_path, dec_path)


def train_dnc(train_env, encoder, decoder, n_iters, dir, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''

    agent = Seq2SeqDNCAgent(train_env, "", encoder, decoder, max_episode_len)
    print('Training with %s feedback' % feedback_method)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    data_log = defaultdict(list)
    start = time.time()

    for idx in range(0, 10000, log_every):

        interval = min(log_every, n_iters - idx)
        iter = idx + interval
        data_log['iteration'].append(iter)
        # Train for log_every interval
        agent.train(encoder_optimizer, decoder_optimizer, interval, feedback=feedback_method, persist=False)

        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR + dir, model_prefix, env_name, iter)
            # Get validation loss under the same conditions as training

            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')
            agent.write_results()
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric, val in score_summary.items():
                data_log['%s %s' % (env_name, metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)
        agent.env = train_env

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                   iter, float(iter) / n_iters * 100, loss_str))

        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s_log.csv' % (PLOT_DIR + dir, model_prefix)
        df.to_csv(df_path)
        #sd = SNAPSHOT_DIR + dir
        #split_string = "-".join(train_env.splits)
        #enc_path = '%s%s_%s_enc_iter_%d' % (sd, model_prefix, split_string, iter)
        #dec_path = '%s%s_%s_dec_iter_%d' % (sd, model_prefix, split_string, iter)
        #agent.save(enc_path, dec_path)


def train_persdnc(train_env, encoder, decoder, n_iters, tgtdir, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''

    agent = Seq2SeqDNCAgent(train_env, "", encoder, decoder, max_episode_len)
    print('Training with %s feedback' % feedback_method)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    data_log = defaultdict(list)
    start = time.time()

    for idx in range(0, n_iters, log_every):



        interval = min(log_every, n_iters - idx)
        iter = idx + interval
        data_log['iteration'].append(iter)
        # Train for log_every interval
        agent.feedback = feedback_method
        agent.train(encoder_optimizer, decoder_optimizer, interval, feedback=feedback_method, persist=True)

        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR + tgtdir, model_prefix, env_name, iter)
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True, persist=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=True, feedback='argmax', allow_cheat=True, persist=True)
            agent.write_results()
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric, val in score_summary.items():
                data_log['%s %s' % (env_name, metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)
        agent.env = train_env

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                   iter, float(iter) / n_iters * 100, loss_str))

        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s_log.csv' % (PLOT_DIR + tgtdir, model_prefix)
        df.to_csv(df_path)
        sd = SNAPSHOT_DIR + tgtdir
        split_string = "-".join(train_env.splits)
        enc_path = '%s%s_%s_enc_iter_%d' % (sd, model_prefix, split_string, iter)
        dec_path = '%s%s_%s_dec_iter_%d' % (sd, model_prefix, split_string, iter)
        agent.save(enc_path, dec_path)


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train', 'val_seen', 'val_unseen']), TRAINVAL_VOCAB)
    print('setup done')


def test_submission():
    ''' Train on combined training and validation sets, and generate test submission. '''

    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train', 'val_seen', 'val_unseen'], tokenizer=tok)

    # Build models and train
    enc_hidden_size = hidden_size // 2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                          dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                              action_embedding_size, hidden_size, dropout_ratio).cuda()
    #train(train_env, encoder, decoder, n_iters)

    # Generate test submission
    test_env = R2RBatch(features, batch_size=batch_size, splits=['test'], tokenizer=tok)
    agent = Seq2SeqAgent(test_env, "", encoder, decoder, max_episode_len)
    agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, 'test', 20000)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()




def train_val_mem(dir='test/', finetune=True):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok)

    # Create directories
    if not os.path.exists(SNAPSHOT_DIR + dir):
        os.makedirs(SNAPSHOT_DIR + dir)
    if not os.path.exists(RESULT_DIR + dir):
        os.makedirs(RESULT_DIR + dir)
    if not os.path.exists(PLOT_DIR + dir):
        os.makedirs(PLOT_DIR + dir)


    # Creat validation environments
    val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
                                 tokenizer=tok), Evaluation([split])) for split in ['val_seen', 'val_unseen']}

    # Build models and train
    enc_hidden_size = hidden_size // 2 if bidirectional else hidden_size

    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                          dropout_ratio, bidirectional=bidirectional)

    if dir == 'C2/':
        decoder = DecoderHistN2N(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                                 action_embedding_size, hidden_size, dropout_ratio)
    else:
        decoder = DecoderHistN2N(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                                     action_embedding_size, hidden_size, dropout_ratio)

    if finetune:
        encoder.load_state_dict(torch.load(SNAPSHOT_DIR +
                                           'seq2seq/seq2seq_sample_imagenet_train_enc_iter_18000'))
        encoder.freeze()
        print('encoder frozen')


    #print(decoder.decoder2action)
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    train_mem(train_env, encoder, decoder, n_iters, dir=dir, val_envs=val_envs)


def train_val_persmem(load_it, srcdir, tgtdir):
    if not os.path.exists(SNAPSHOT_DIR + srcdir):
        print('wrong path')
        return
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok)

    if not os.path.exists(SNAPSHOT_DIR+ tgtdir):
        os.makedirs(SNAPSHOT_DIR + tgtdir)
        os.makedirs(RESULT_DIR + tgtdir)
        os.makedirs(PLOT_DIR + tgtdir)

    val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
                                 tokenizer=tok), Evaluation([split])) for split in ['val_seen', 'val_unseen']}
    # Build models and train
    enc_hidden_size = hidden_size // 2 if bidirectional else hidden_size

    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                          dropout_ratio, bidirectional=bidirectional)
    if srcdir == 'C2/':
        decoder = DecoderHistN2N(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                                 action_embedding_size, hidden_size, dropout_ratio)
    else:
        decoder = DecoderHistN2NAttn(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                                     action_embedding_size, hidden_size, dropout_ratio)
    encoder.load_state_dict(torch.load(SNAPSHOT_DIR + srcdir + '/seq2seq_sample_train_enc_iter_' + str(load_it)))
    decoder.load_state_dict(torch.load(SNAPSHOT_DIR + srcdir + '/seq2seq_sample_train_dec_iter_' + str(load_it)))
    encoder = encoder.cuda(0)
    decoder = decoder.cuda(0)
    encoder.freeze()
    decoder.freeze()
    train_persmem(train_env, encoder, decoder, n_iters, tgtdir, val_envs=val_envs)


def train_val_dnc(rh, mem_embed_size, dir, finetune=True):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok)

    # Create directories
    if not os.path.exists(SNAPSHOT_DIR + dir):
        os.makedirs(SNAPSHOT_DIR + dir)
        os.makedirs(RESULT_DIR + dir)
        os.makedirs(PLOT_DIR + dir)


    # Creat validation environments
    val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
                                 tokenizer=tok), Evaluation([split])) for split in ['val_seen', 'val_unseen']}

    # Build models and train
    enc_hidden_size = hidden_size // 2 if bidirectional else hidden_size

    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                          dropout_ratio, bidirectional=bidirectional)
    decoder = DecoderSingleDNC(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(), action_embedding_size,
                               hidden_size, dropout_ratio, mem_embed_size=mem_embed_size, rh=rh)
    if finetune:
        encoder.load_state_dict(torch.load(SNAPSHOT_DIR +
                                           'seq2seq/seq2seq_sample_imagenet_train_enc_iter_18000'))
        encoder.freeze()

    encoder = encoder.cuda(0)
    decoder = decoder.cuda(0)
    train_dnc(train_env, encoder, decoder, n_iters, dir=dir, val_envs=val_envs)



def train_val_persdnc(load_it, srcdir, tgtdir):
    if not os.path.exists(SNAPSHOT_DIR + srcdir):
        print('wrong path')
        return
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok)

    if not os.path.exists(SNAPSHOT_DIR+ tgtdir):
        os.makedirs(SNAPSHOT_DIR + tgtdir)
        os.makedirs(RESULT_DIR + tgtdir)
        os.makedirs(PLOT_DIR + tgtdir)

    val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
                                 tokenizer=tok), Evaluation([split])) for split in ['val_seen', 'val_unseen']}
    # Build models and train
    enc_hidden_size = hidden_size // 2 if bidirectional else hidden_size

    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                          dropout_ratio, bidirectional=bidirectional)
    decoder = DecoderSingleDNC(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                             action_embedding_size, hidden_size, dropout_ratio)
    encoder.load_state_dict(torch.load(SNAPSHOT_DIR + srcdir + '/seq2seq_sample_train_enc_iter_' + str(load_it)))
    decoder.load_state_dict(torch.load(SNAPSHOT_DIR + srcdir + '/seq2seq_sample_train_dec_iter_' + str(load_it)))
    encoder = encoder.cuda(0)
    decoder = decoder.cuda(0)
    encoder.freeze()
    #decoder.freeze()
    train_persdnc(train_env, encoder, decoder, n_iters, tgtdir, val_envs=val_envs)




if __name__ == "__main__":
    # get_embeddings()
    #train_val()
    #train_val_mem(dir='test/', finetune=True)
    #train_val_mem(dir='C_attn/', finetune=True)

    #train_val_persmem(load_it=19800, srcdir='C2/', tgtdir='test/')
    #train_val_persmem(load_it=15200, srcdir='C_attn/', tgtdir='test/')
   # train_val_dnc(rh=1, mem_embed_size=1500, dir='test/')
    #train_val_dnc(rh=2, mem_embed_size=2000, dir='dnc_2_2000/')
    #train_val_dnc(rh=3, mem_embed_size=2000, dir='dnc_3_2000/')


    train_val_persdnc(load_it=6100, srcdir='dnc/', tgtdir='test/')

    #train_dnc(2, 500)
    # test_submission()
