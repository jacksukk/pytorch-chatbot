import torch
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
from tqdm import tqdm
import math as m
import argparse
import string
from numpy import dot
from numpy.linalg import norm

def parseFilename(filename, test=False):
    filename = filename.split('/')
    dataType = filename[-1][:-4] # remove '.tar'
    parse = dataType.split('_')
    reverse = 'reverse' in parse
    layers, hidden = filename[-2].split('_')
    n_layers = int(layers.split('-')[0])
    hidden_size = int(hidden)
    return n_layers, hidden_size, reverse

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 設定 Keras 使用的 TensorFlow Session
tf.keras.backend.set_session(sess)
model = word2vec.Word2Vec.load('word2vec.model')
model2 = load_model("model.h5")
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
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
def decode(decoder, decoder_hidden, encoder_outputs, voc, sentence):
    sentence = sentence.split(' ')
    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input = decoder_input.to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(15, 15) #TODO: or (MAX_LEN+1, MAX_LEN+1)
    loss = 0
    for di in range(len(sentence)):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        ni = voc.word2index[sentence[di]]
        t = torch.tensor([ni]).to(device)
      #  print(decoder_output[0])
        loss += F.cross_entropy(decoder_output[0].unsqueeze(0), t, ignore_index=EOS_token)
        decoder_input = torch.LongTensor([[ni]])
        decoder_input = decoder_input.to(device)

    return loss.item() / len(sentence)
def perplexity(encoder, decoder, voc, sentence):
    question = sentence[0]
    indexes_batch = [indexesFromSentence(voc, question)] #[1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)

    encoder_outputs, encoder_hidden = encoder(input_batch, lengths, None)

    decoder_hidden = encoder_hidden[:decoder.n_layers]
    return decode(decoder, decoder_hidden, encoder_outputs, voc, sentence[1])
def correlation(encoder, voc, sentence):
    question = sentence[0]
    indexes_batch = [indexesFromSentence(voc, question)] #[1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    encoder_outputs, encoder_hidden_1 = encoder(input_batch, lengths, None)
    question = sentence[1]
    indexes_batch = [indexesFromSentence(voc, question)] #[1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    encoder_outputs, encoder_hidden_2 = encoder(input_batch, lengths, None)
    a = encoder_hidden_1[-1][0].cpu().detach().numpy()
    b = encoder_hidden_2[-1][0].cpu().detach().numpy()

    return dot(a, b) / norm(a) / norm(b) 
def main():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-o', '--output', help='outputfile')
    args = parser.parse_args()
    n_layers, hidden_size, reverse = parseFilename(args.test, True)
    torch.set_grad_enabled(False)
    voc, pairs = loadPrepareData('data/movie_subtitles.txt')
    embedding = nn.Embedding(voc.n_words, hidden_size)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)
    checkpoint = torch.load('./50000_backup_bidir_model.tar')
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    encoder.train(False);
    decoder.train(False);
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    total_loss = 0
    total_score = 0
    count = 0
    with open(args.output) as f:
        ans = f.readlines()
    temp = []
    test_score = []
    corr1 = 0
    corr2 = 0
    for sentence in tqdm(ans):
        sentence = sentence[:-1]
        if sentence != '=========================':
            temp.append(sentence)
        else:
            total_loss += perplexity(encoder, decoder, voc, temp[:2])
            corr1 += correlation(encoder, voc, temp[:2])
            corr2 += correlation(encoder, voc, [temp[0], temp[-1]])
            test_score.append(temp[-1].lower().translate(str.maketrans(' ', ' ', string.punctuation)).split())
            temp = []
    print('perplexity:', m.pow(2, total_loss/len(test_score)))
    print('correlation_in_1:', corr1 / len(test_score))
    print('correlation_in_2:', corr2 / len(test_score))
    print('scroe:', sum(test(test_score))/len(test_score))


if __name__ == '__main__':
    main()