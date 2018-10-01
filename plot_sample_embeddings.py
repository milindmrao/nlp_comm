import os
import sys
import tensorflow as tf
import numpy as np
from SentenceBatchGenerator import SentenceBatchGenerator, Word2Numb
from EncDecChanModels import Config
import nltk
from SentenceEncChanDecNet import BeamSearchEncChanDecNet
from sklearn import manifold
import matplotlib.pyplot as plt

if __name__ == '__main__':
    chan_params = {"type": "erasure", "keep_prob": 0.95}

    parent_dir, _ = os.path.split(os.getcwd())
    print(parent_dir)

    print('Init and Loading Data...')
    config = Config(None,chan_params,numb_tx_bits=400, lr=0.001, peephole=True, batch_size=1)
    word2numb = Word2Numb(config.w2n_path)

    modelPath = "d0.05-b400"
    model_save_path = os.path.join(parent_dir, 'trained_models', modelPath)
    
    test_sentences = ['The car is coming',
                      'An automobile is arriving',
                      'A vehicle is approaching',
                      'A car will drive by',
                      'The girl told him that',
                      'The woman said that to him',
                      'The girl declared it to him',
                      'A female person uttered it',
                      'politicians have voted',
                       'few politicians decided',
                       'Politicians have elected',
                       'the politicians have chosen']
    
    word_token = nltk.tokenize.WordPunctTokenizer()
    def tokenizer(sentence):
        words = word_token.tokenize(sentence)
        words = [w.lower() for w in words]
        tokens = word2numb.convert_w2n(words)
        return tokens
    
    tokens = [tokenizer(x) for x in test_sentences]


    beam_sys = BeamSearchEncChanDecNet(config, word2numb, beam_size=1)

    print('Start session...')
    with tf.Session() as sess:
        beam_sys.load_enc_dec_weights(sess, model_save_path)
        encoding = np.zeros([len(test_sentences),400])
        for ind,token in enumerate(tokens):
            chan_out = beam_sys.encode_Tx_sentence(sess,token,keep_rate=1)
            encoding[ind,:]= np.reshape(chan_out,[1,-1]) 
            
    distance_matrix = np.array([[np.linalg.norm(x-y) for x in encoding] for y in encoding])
    manifold_fitter = manifold.MDS(dissimilarity='precomputed')
    coordinates = manifold_fitter.fit_transform(distance_matrix)
    x_coord,y_coord = coordinates[:,0],coordinates[:,1]
    
    plt.figure(1)
    markers = ['o']*4+['v']*4+['s']*4
    colors = ['k']*4+['r']*4+['g']*4
    for x_c,y_c,mark,col,test_sentence in zip(coordinates[:,0],coordinates[:,1],markers,colors,test_sentences):
        plt.scatter(x_c,y_c,c=col,marker=mark)
        plt.annotate(xy=(x_c,y_c),s=test_sentence)
    plt.xlim([-20,40])
    plt.savefig(os.path.join(parent_dir,'results','example_embedding.eps'))