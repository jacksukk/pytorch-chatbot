import torch
import random
random.seed(4)
#from train import indexesFromSentence
from load import SOS_token, EOS_token
from load import MAX_LENGTH, loadPrepareData, Voc
from model import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
from keras import backend as K                                       
from keras.preprocessing.sequence import pad_sequences                 
from keras.models import load_model
import tensorflow as tf
import math
import time                                                                                                                                                        
from gensim.models import word2vec                                   
import pandas as pd                                                  
import numpy as np   

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


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc.index2word[idx.item()] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc.index2word[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())

def beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for i in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = torch.LongTensor([[sentence.last_idx]])
            decoder_input = decoder_input.to(device)

            decoder_hidden = sentence.decoder_hidden
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []

    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]

def decode(decoder, decoder_hidden, encoder_outputs, voc, max_length=MAX_LENGTH):

    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input = decoder_input.to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length) #TODO: or (MAX_LEN+1, MAX_LEN+1)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        _, topi = decoder_output.topk(3)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc.index2word[ni.item()])

        decoder_input = torch.LongTensor([[ni]])
        decoder_input = decoder_input.to(device)

    return decoded_words, decoder_attentions[:di + 1]


def evaluate(encoder, decoder, voc, sentence, beam_size, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)

    encoder_outputs, encoder_hidden = encoder(input_batch, lengths, None)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    if beam_size == 1:
        return decode(decoder, decoder_hidden, encoder_outputs, voc)
    else:
        return beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size)


def evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, testenco, testdeco, n=10):
    for _ in range(n):
        pair = random.choice(pairs)
        print("=============================================================")
        if reverse:
            print('>', " ".join(reversed(pair[0].split())))
        else:
            print('>', pair[0])
        if beam_size == 1:

            import string
            output_words, _ = evaluate(encoder, decoder, voc, pair[0], beam_size)
            output_words = output_words[:-1]
            output_sentence = ' '.join(output_words)
            recieve, _ = evaluate(testenco, testdeco, voc, output_sentence, beam_size)
            print('<', output_sentence)
            recieve = recieve[:-1]
            recieve[-1] = recieve[-1][:-1]
            print('<>', ' '.join(recieve))
            temp = temp.lower().translate(str.maketrans(' ', ' ', string.punctuation))
            recieve = temp.split()
            score = test(recieve)
            print(score[0][0].item())
        else:
            output_words_list = evaluate(encoder, decoder, voc, pair[0], beam_size)
            for output_words, score in output_words_list:
                output_sentence = ' '.join(output_words)
                print("{:.3f} < {}".format(score, output_sentence))

def evaluateInput(encoder, decoder, voc, beam_size):
    pair = ''
    while(1):
        try:
            pair = input('> ')
            if pair == 'q': break
            if beam_size == 1:
                output_words, _ = evaluate(encoder, decoder, voc, pair, beam_size)
                output_sentence = ' '.join(output_words)
                print('<', output_sentence)
            else:
                output_words_list = evaluate(encoder, decoder, voc, pair, beam_size)
                for output_words, score in output_words_list:
                    output_sentence = ' '.join(output_words)
                    print("{:.3f} < {}".format(score, output_sentence))
        except KeyError:
            print("Incorrect spelling.")


def runTest(n_layers, hidden_size, reverse, modelFile, beam_size, inp, corpus):
    torch.set_grad_enabled(False)

    voc, pairs = loadPrepareData(corpus)
    embedding = nn.Embedding(voc.n_words, hidden_size)
    embedding2 = nn.Embedding(voc.n_words, hidden_size)
    testenco = EncoderRNN(voc.n_words, hidden_size, embedding2, n_layers)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)
    testdeco = LuongAttnDecoderRNN(attn_model, embedding2, hidden_size, voc.n_words, n_layers)
    checkpoint = torch.load(modelFile)
    checkpoint2 = torch.load('./50000_backup_bidir_model.tar')
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    testenco.load_state_dict(checkpoint2['en'])
    testdeco.load_state_dict(checkpoint2['de'])

    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False);
    decoder.train(False);

    testenco.eval()
    testdeco.eval()
    testenco = testenco.to(device)
    testdeco = testdeco.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if inp:
        evaluateInput(encoder, decoder, voc, beam_size)
    else:
        evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, testenco, testdeco, 10)
