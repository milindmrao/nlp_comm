# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:29:45 2017

@author: Milind

preprocessing word files. read sentences. 1) optional Form batches of nearly the same size. 
"""

import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import nltk
import pickle

""" creating a file with a sentence in a new line and sorting based on sentence
length for the reuters and qasys datasets """

PAD_ID = 0
START_ID = 1
END_ID = 2
UNK_ID = 3


def combine_input(file_name, params={}):
    """ creates file from Reuters dataset. Prints
    Input:
        file_name - training or test
        params - output_file_name, min_length,max_length
    """

    # good sentence tokenizer - nltk.tokenize.PunktSentenceTokenizer()
    # good word tokenizer - nltk.tokenize.WordPunktTokenizer()
    word_token = nltk.tokenize.WordPunctTokenizer()
    sent_token = nltk.tokenize.PunktSentenceTokenizer()
    parent_dir, _ = os.path.split(os.getcwd())
    data_dir = os.path.join(parent_dir, 'data', 'corpora', 'reuters', file_name)
    sentences = list()
    #    lens = list()
    min_length = params.get('min_length', 0)
    max_length = params.get('max_length', 1000)

    for fname in tqdm(os.listdir(data_dir)):
        with open(os.path.join(data_dir, fname), 'r') as fop:
            fop.readline()
            text_raw = fop.read()
            sents = sent_token.tokenize(text_raw)

            def f_rest(x):
                _ = len(word_token.tokenize(x))
                return _ >= min_length and _ <= max_length

            sents_legal = list(filter(f_rest, sents))
            sentences += sents_legal
            #            lens += [len(nltk.tokenize.word_tokenize(x)) for x in sents_legal]

    # We need to club sentences of the same length to improve the efficiency of gradient descent.
    output_file_name = params.get('output_file_name', file_name)
    with open(os.path.join(parent_dir, 'data', output_file_name + '_reuters.pickle'), 'wb') as fop:
        pickle.dump(sentences, fop)
    return sentences


def process_sentence_eur():
    parent_dir = os.path.split(os.getcwd())[0]
    parent_dir, _ = os.path.split(parent_dir)
    path_to_eur = os.path.join(parent_dir, 'data', 'corpora', 'europarl-v7.en', 'europarl-v7.en')
    sentences = []
    word_token = nltk.tokenize.WordPunctTokenizer()
    with open(path_to_eur, 'r', encoding='utf-8') as fop:
        for line in tqdm(fop.readlines()):
            _len_line = len(word_token.tokenize(line))
            if _len_line >= 4 and _len_line <= 40:
                sentences += [line]

    target_path = os.path.join(parent_dir, 'data', 'training_euro.pickle')
    pickle.dump(sentences, open(target_path, 'wb'))


def word_to_vec(length=50000, dim=50):
    """Aim of this is to create the vocabulary and the embeddings"""
    parent_dir, _ = os.path.split(os.getcwd())
    parent_dir, _ = os.path.split(parent_dir)
    words_special = [('<pad>', PAD_ID), ('<start>', START_ID), ('<end>', END_ID), ('<unk>', UNK_ID)]

    words = []
    embedding = [list(np.random.randn(dim)) for x in range(len(words_special))]
    embedding[0] = list(np.zeros(dim))
    with open(os.path.join(parent_dir, 'data', 'glove.6b.' + str(dim) + 'd.txt'), 'r', encoding="utf8") as fop:
        for ind in tqdm(range(length)):
            #            try:
            #                _ = fop.readline()
            #            except:
            #                print(ind)
            #                print(_)
            #                print(fop.readline())
            #                print(words[-20:],embedding[-2:])
            #                return
            line = fop.readline().split()
            words += [line[0]]
            embedding += [list(map(float, line[1:]))]
    word2num = dict(words_special + list(zip(words, range(len(words_special), len(words) + len(words_special)))))
    num2word = dict([(n, w) for w, n in word2num.items()])
    embedding = np.array(embedding)
    with open(os.path.join(parent_dir, 'data', str(dim) + '_embed_large.pickle'), 'wb') as fop:
        pickle.dump(embedding, fop)

    with open(os.path.join(parent_dir, 'data', 'w2n_n2w.pickle'), 'wb') as fop:
        pickle.dump([word2num, num2word], fop)


def test_w2n():
    parent_dir, _ = os.path.split(os.getcwd())
    with open(os.path.join(parent_dir, 'data', 'w2n_n2w.pickle'), 'rb') as fop:
        [w2n, n2w] = pickle.load(fop)
    return [w2n, n2w]


if __name__ == "__main__":
    # combine_input('test',{'output_file_name':'test_rest','min_length':4,'max_length':50})
    # combine_input('test')
    # word_to_vec(200000,100)
    word_to_vec()
    sentences = process_sentence_eur()
    pass