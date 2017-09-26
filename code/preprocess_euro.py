import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import nltk
import pickle
import random

""" creating a file with a sentence in a new line and sorting based on sentence
length for the reuters and qasys datasets """



PAD_ID = 0
END_ID = 1
START_ID = 2
UNK_ID = 3



def process_sentence_eur(test_percent):
    parent_dir = os.path.split(os.getcwd())[0]
    parent_dir, _ = os.path.split(parent_dir)
    path_to_eur = os.path.join(parent_dir, 'data', 'corpora', 'europarl-v7.en', 'europarl-v7.en')
    #path_to_eur = os.path.join(parent_dir, 'data', 'corpora', 'europarl-v7.en', 'small_test_example.txt')
    sentences = []
    word_token = nltk.tokenize.WordPunctTokenizer()
    word_freq = nltk.FreqDist()
    with open(path_to_eur, 'r', encoding='utf-8') as fop:
        for line in tqdm(fop.readlines()):
            words = word_token.tokenize(line)
            words = [w.lower() for w in words]
            word_freq.update(words)
            #print(words)
            #_len_line = len(words)
            #if _len_line >= 4 and _len_line <= 20:
                #sentences.append(words)

    #len_sent = len(sentences)
    #train_endIdx = round(len_sent*test_percent)
    #random.shuffle(sentences)
    #test_sent = sentences[0:train_endIdx]
    #train_sent = sentences[train_endIdx:]

    #train_path = os.path.join(parent_dir, 'data', 'training_euro_wordlist20.pickle')
    #test_path = os.path.join(parent_dir, 'data', 'testing_euro_wordlist20.pickle')
    #pickle.dump(train_sent, open(train_path, 'wb'))
    #pickle.dump(test_sent, open(test_path, 'wb'))
    return word_freq

def word_to_vec(common_words, dim=50):
    """Aim of this is to create the vocabulary and the embeddings"""
    parent_dir, _ = os.path.split(os.getcwd())
    parent_dir, _ = os.path.split(parent_dir)
    words_special = [('<pad>', PAD_ID), ('<end>', END_ID), ('<start>', START_ID), ('<unk>', UNK_ID)]

    words = []
    embedding = [list(np.random.randn(dim)) for x in range(len(words_special))]
    embedding[0] = list(np.zeros(dim))
    #with open(os.path.join(parent_dir, 'data', 'glove.6b.' + str(dim) + 'd.txt'), 'r', encoding="utf8") as fop:
    with open(os.path.join(parent_dir, 'data', 'glove.6b.' + str(dim) + 'd.txt'), 'r', encoding="utf8") as fop:
        for line in tqdm(fop.readlines()):
            split_line = line.split()
            #print(line)
            if common_words.get(split_line[0],0)>0:
                words += [split_line[0]]
                embedding += [list(map(float, split_line[1:]))]
    word2num = dict(words_special + list(zip(words, range(len(words_special), len(words) + len(words_special)))))
    #print(word2num)
    num2word = dict([(n, w) for w, n in word2num.items()])
    embedding = np.array(embedding)
    #print(embedding)
    with open(os.path.join(parent_dir, 'data', str(dim) + '_embed_large_TopEuro.pickle'), 'wb') as fop:
        pickle.dump(embedding, fop)

    with open(os.path.join(parent_dir, 'data', 'w2n_n2w_TopEuro.pickle'), 'wb') as fop:
        pickle.dump([word2num, num2word], fop)


if __name__ == "__main__":
    word_freq = process_sentence_eur(0.15)
    #print("total words: ", len(word_freq))
    common_words = dict(word_freq.most_common(20000))
    #print("got common words")
    word_to_vec(common_words, dim=100)
    word_to_vec(common_words, dim=200)
    word_to_vec(common_words, dim=300)
    pass