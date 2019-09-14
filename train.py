import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.backends.cudnn as cudnn
import numpy as np
import itertools
import random
import math
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token
from model import EncoderRNN, LuongAttnDecoderRNN
from config import MAX_LENGTH, teacher_forcing_ratio, save_dir
from evaluate import evaluate
#from twitter.test_sentence import test
import math
import time                                                                                                                                                        
from gensim.models import word2vec                                   
import pandas as pd                                                  
import numpy as np                                                   
from keras import backend as K                                       
from keras.preprocessing.sequence import pad_sequences                 
from keras.models import load_model
import tensorflow as tf

# 只使用 30% 的 GPU 記憶體
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 設定 Keras 使用的 TensorFlow Session
tf.keras.backend.set_session(sess)
model = word2vec.Word2Vec.load('word2vec.model')
model2 = load_model("model.h5")

def test(sentences):
    
    test_x = []
    for sentence in sentences:
        temp = []
        for w in sentence:
            if w not in model:
                temp.append(np.zeros(256))
                continue
            temp.append(model[w])
        test_x.append(temp[:])
    if test_x == []:
        test_x = [[np.zeros(256)]]
    test_x = pad_sequences(test_x, maxlen=48, padding='post', truncating='post', value=np.zeros(256))


    y_predict = model2.predict(test_x)
    return y_predict


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

cudnn.benchmark = True
#############################################
# generate file name for saving parameters
#############################################
def filename(reverse, obj):
	filename = ''
	if reverse:
		filename += 'reverse_'
	filename += obj
	return filename


#############################################
# Prepare Training Data
#############################################
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# batch_first: true -> false, i.e. shape: seq_len * batch
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# convert to index, add EOS
# return input pack_padded_sequence
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = [len(indexes) for indexes in indexes_batch]
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# convert to index, add EOS, zero padding
# return output variable, mask, max length of the sentences in batch
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by input length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(voc, pair_batch, reverse):
    if reverse:
        pair_batch = [pair[::-1] for pair in pair_batch]
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

#############################################
# Training
#############################################
def criterion(state_prob, episode_reward):
    loss = torch.zeros(1).to(device)
    for p in state_prob:
        loss += - torch.log(p) * episode_reward.item()
    return loss
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, testenco, testdeco, voc, batch_size, iteration, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)


    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = False
    # Run through decoder one time step at a time
    sentence = [[] for i in range(batch_size)]
    state_prob = [[] for i in range(batch_size)]
    state_loss = [0 for i in range(batch_size)]
   # normal_loss = [0 for i in range(batch_size)]
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topk, topi = decoder_output.topk(1) # [64, 1]
            sample = torch.multinomial(decoder_output, 1) #(64, 1)
            flag = True
            for i in range(batch_size):
                if voc.index2word[sample[i][0].item()] == 'EOS' or voc.index2word[sample[i][0].item()] == 'PAD':
                    continue
                state_prob[i].append(decoder_output[i][sample[i][0]])
                sentence[i].append(voc.index2word[sample[i][0].item()])
            
            #sentence.append(voc.index2word[ni.item()])
            decoder_input = target_variable[t].view(1, -1) # Next input is current target
#loss += F.cross_entropy(decoder_output, target_variable[t], ignore_index=EOS_token)
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            #print(decoder_output)
            topk, topi = decoder_output.topk(1) # [64, 1]
            #print(decoder_output)
            randomone = torch.ones([64, 92383]).to(device)
           # if iteration > 1000:
            sample = torch.multinomial(F.softmax(decoder_output, dim=1), 1)
            #else:
            #    sample = torch.multinomial(randomone, 1) #(64, 1)
          #  print(sample)
          #  ni = topi[0][0]
           # print(ni)
            #print(topk[0][0].item(), decoder_output[0][ni])
           # print(sample.view(-1))
            flag = True
#print(decoder_output.shape)
            for i in range(batch_size):
                if voc.index2word[sample[i][0].item()] == 'EOS' or voc.index2word[sample[i][0].item()] == 'PAD':
                    continue
            #    state_prob[i].append(decoder_output[i][sample[i][0]])
                sentence[i].append(voc.index2word[sample[i][0].item()])
                state_loss[i] = state_loss[i] + F.cross_entropy(decoder_output[i].unsqueeze(0), sample.view(-1)[i].unsqueeze(0), ignore_index=EOS_token)
            #sentence.append(voc.index2word[ni.item()])
            decoder_input = torch.LongTensor([[sample[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            loss += F.cross_entropy(decoder_output, target_variable[t], ignore_index=EOS_token)
            #loss += F.cross_entropy(decoder_output, sample.view(-1), ignore_index=EOS_token)

    #sentence = ' '.join(sentence) 
#    start_time = time.time()
    for i in range(len(sentence)):
        temp = ' '.join(sentence[i])
        if i == 0:
            print('> ', temp)
        output_words, _ = evaluate(testenco, testdeco, voc, temp, 1)
        output_words = output_words[:-1]
        output_words[-1] = output_words[-1][:-1]
        sentence[i] = output_words[:]
 #   elapsed_time = time.time() - start_time
#    print("evaluate", elapsed_time)
  #  start_time = time.time()
# your code
    score = test(sentence)
#    elapsed_time = time.time() - start_time
#    print("test take time", elapsed_time)
 #   print(score)
    #loss = loss * score[0][0].item()
    for i in range(batch_size):
        loss += state_loss[i] * score[i][0].item()
#    loss = loss * score.item()
    # for i in range(batch_size):
    #     loss += criterion(state_prob[i], score[i][0])
    #score = [[5]]
   # loss += criterion(state_prob, score[0][0])

    loss.backward()
    print('< ', ' '.join(output_words))
    print("Reward:%.2f,     loss:%.6f"%(np.mean(score), loss))
    clip = 50.0

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / max_target_len 


def trainIters(corpus, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size,
                print_every, save_every, dropout, loadFilename=None, attn_model='dot', decoder_learning_ratio=5.0):

    voc, pairs = loadPrepareData(corpus)

    # training data
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    training_batches = None
    try:
        training_batches = torch.load(os.path.join(save_dir, 'training_data', corpus_name,
                                                   '{}_{}_{}.tar'.format(n_iteration, \
                                                                         filename(reverse, 'training_batches'), \
                                                                         batch_size)))
    except FileNotFoundError:
        print('Training pairs not found, generating ...')
        training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], reverse)
                          for _ in range(n_iteration)]
        torch.save(training_batches, os.path.join(save_dir, 'training_data', corpus_name,
                                                  '{}_{}_{}.tar'.format(n_iteration, \
                                                                        filename(reverse, 'training_batches'), \
                                                                        batch_size)))
    # model
    checkpoint = None
    print('Building encoder and decoder ...')
    embedding = nn.Embedding(voc.n_words, hidden_size)
    embedding2 = nn.Embedding(voc.n_words, hidden_size)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers, dropout)
    testenco = EncoderRNN(voc.n_words, hidden_size, embedding2, n_layers, dropout)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers, dropout)
    testdeco = LuongAttnDecoderRNN(attn_model, embedding2, hidden_size, voc.n_words, n_layers, dropout)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
        testenco.load_state_dict(checkpoint['en'])
        testdeco.load_state_dict(checkpoint['de'])
    # use cuda
    encoder.train(True)
    decoder.train(True)
    testenco.eval()
    testdeco.eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    testenco = testenco.to(device)
    testdeco = testdeco.to(device)

    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    # initialize
    print('Initializing ...')
    start_iteration = 1
    perplexity = []
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
        perplexity = checkpoint['plt']
    start_iteration = 0
    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, testenco, testdeco, voc, batch_size, iteration)
        print_loss += loss
        perplexity.append(loss)

        if iteration % print_every == 0:
            print_loss_avg = math.exp(print_loss / print_every)
            print('%d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, 'model', corpus_name, '{}-{}_{}'.format(n_layers, n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, filename(reverse, 'backup_bidir_model'))))
