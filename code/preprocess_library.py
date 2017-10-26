# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:43:13 2017

This is the library of preprocessing functions

@author: Milind, Nariman
"""

import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import nltk
import pickle
import random
from collections import deque
import bisect
import itertools


# Pre-initializing what the constants should be
PAD_ID = 0
END_ID = 1
START_ID = 2
UNK_ID = 3

def word_dict_embed(vocab_size=50000, **kwargs):
    """This function creates the word2num, num2word and embeddings objects. It 
    first goes through the corpus to extract word counts. It then uses these and 
    a pretrained embeddings file to 
    
    Args:
        vocab_size: int. default 50000
        dim: int. Dimensionality of word embeddings. Default 50
        path_corpus: path of the text corpora
        path_embed: path of the embeddings file. Default is constructed from dim
           size
        path_w2n_n2w: path to save the w2n_n2w file
        path_word_embed: path to save the final word embeddings
        iter_limit: maximum number of lines to read in the corpus file.
    Returns:
        
    """
    parent_dir = os.path.split(os.getcwd())[0]
    dim = kwargs.get('dim',200)
    path_to_eur = os.path.join(parent_dir, 'data', 'corpora', 'europarl-v7.en', 'europarl-v7.en')
    path_corpus = kwargs.get('path_corpus',path_to_eur)
    path_glove_embed = os.path.join(parent_dir,'data','glove.6b.%dd.txt'%dim)
    path_embed = kwargs.get('path_embed',path_glove_embed)
    path_w2n_n2w = kwargs.get('path_w2n_n2w',
                              os.path.join(parent_dir, 'data', 'w2n_n2w_euro.pickle'))
    path_word_embed = kwargs.get('path_word_embed',
                                 os.path.join(parent_dir, 'data', '%d_embed_euro.pickle'%dim))
    iter_limit = kwargs.get('iter_limit',10000000)
    #------- Extracting the most common words from the corpus-----------
    word_token = nltk.tokenize.WordPunctTokenizer()
    word_freq = nltk.FreqDist()
    with open(path_corpus,'r',encoding='utf8') as fop:
        print('Reading the corpus to extract word frequencies')
        for line in tqdm(itertools.islice(fop,0,iter_limit)):
            words = (w.lower() for w in 
                     word_token.tokenize(line[:-1])) #Removing the newline, extract token, convert to lower case
            word_freq.update(words)
    
    #-------- using word frequencies to create word2num, num2word -------
    words_special = [('<pad>', PAD_ID), ('<end>', END_ID), ('<start>', START_ID), ('<unk>', UNK_ID)]
    len_words_special = len(words_special)
    common_words,_ = zip(*word_freq.most_common(vocab_size))
    common_words_index = zip(common_words,range(len_words_special,len_words_special + len(common_words)))
    words_index = words_special + list(common_words_index)
    
    word2num = dict(words_index)
    num2word = dict([n,w] for (w,n) in word2num.items())
    
    with open(path_w2n_n2w, 'wb') as fop:
        pickle.dump([word2num, num2word], fop)
        
    #-------- Initializing word embeddings -------------------------------
    embedding = np.zeros([len(word2num),dim])
    not_present_word = set(range(len(word2num)))
    with open(path_embed,'r',encoding='utf8') as fop:
        print('Saving pre-trained embeddings')
        for line in tqdm(fop):
            split_line = line[:-1].split()
            word = word2num.get(split_line[0],False)
            if word:
                not_present_word.discard(word)
                embedding[word,:] = list(map(float, split_line[1:]))
    
    embedding[list(not_present_word),:] = np.random.uniform(-0.7,0.7,[len(not_present_word),dim])
    with open(path_word_embed, 'wb') as fop:
        pickle.dump(embedding, fop)
        
    return word2num,num2word,embedding

class Word2Numb(object):
    def __init__(self, w2n_path):
        # ==== load num2word and word2num =======
        with open(w2n_path, 'rb') as fop:
            [self.w2n, self.n2w] = pickle.load(fop)
        print('loaded dictionary of size ',len(self.w2n))
        self.UNK_ID = self.w2n['<unk>']

    def convert_w2n(self, sentence):
        return [self.w2n.get(x, self.UNK_ID) for x in sentence]

    def convert_n2w(self, numbs):
        return [self.n2w.get(x, "<>") for x in numbs]

class SentenceBatchGenerator (object):
    def __init__(self, corp_path,
                 word2numb,
                 **kwargs):
        """ 
        Args are moved into kwargs for backwards compatibility
        Args:
            corp_path: path to the corpus - could be file or folder. Currently assume each line is a new sentence
            word2numb: word2numb object 
            batch_size: size of each batch
            min_len: minimum length of the queue
            max_len: maximum length of the queue
            diff: the difference between sentence length in each batch
            epochs: the number of epochs
            unk_perc: the unknown word percentage
        """

        self.corp_path = corp_path
        self.word2numb = word2numb
        self.batch_size = kwargs.get('batch_size',32)
        self.UNK_ID = kwargs.get('UNK_ID',word2numb.w2n['<unk>'])
        self.unk_perc = kwargs.get('unk_perc',0.2)
        self.epochs = kwargs.get('epochs',1)
        self.curr_epoch = 0
        self.file_pointer = open(corp_path,'r',encoding='utf8')
        self.min_len = kwargs.get('min_len',4)
        self.max_len = kwargs.get('max_len',30)
        diff = kwargs.get('diff',4)
        self.init_batch_queues(self.min_len,self.max_len,diff)
        self._word_tokenizer = nltk.tokenize.WordPunctTokenizer()
        self.do_not_fill = set([]) #List of elements of the queue to not fill anymore

    def init_batch_queues(self,min_len,max_len,diff):
        self.batch_queues = []
        self.queue_limits = list(range(min_len,max_len,diff)) + [max_len]
        self.numb_queues = len(self.queue_limits)-1
        self.batch_queues = [deque() for _ in self.queue_limits[:-1]]

    def prepare_batch_queues(self):
        """ Function does a hard reset of the file pointer"""
        self.file_pointer = open(self.corp_path,'r',encoding='utf8')
        
    def update_do_not_fill(self,index):
        """ Add index to the list of elements of do_not_fill
        """
        self.do_not_fill.add(index)
        
    def fill_batch_queues(self, num_lines_read=100, randomize=True):
        """ This function reads in num_lines_read number of lines. 
        It then converts it numbers, does filtering and places it in the appropriate batch
        
        Args:
            num_lines_read: number of lines to read at once
            randomize: whether to randomize read lines
        """
        for sentence in itertools.islice(self.file_pointer,num_lines_read):
            sentence = sentence.lower()[:-1]
            words = self._word_tokenizer.tokenize(sentence)
            if len(words)<self.min_len or len(words)>self.max_len:
                continue
            words_nums = self.word2numb.convert_w2n(words)
            if sum([w==self.UNK_ID for w in words_nums])/len(words) > self.unk_perc:
                continue
            idx =  bisect.bisect(self.queue_limits,len(words))-1
            if idx==self.numb_queues:
                idx = self.numb_queues-1
            if idx not in self.do_not_fill:
                self.batch_queues[idx].appendleft(sentence)
            if len(self.do_not_fill) == self.numb_queues:
                raise ValueError('no more queues can serve')
            
        
    def can_serve(self):
        """ Returns the id of the queue which is in a position to serve"""
        idxes = list(filter(
                lambda x: len(self.batch_queues[x])>=self.batch_size and x not in self.do_not_fill,
                              range(self.numb_queues)))
        if idxes:
            return idxes[0]
        else:
            return None

    def get_next_batch(self,randomize=True,do_not_fill = False):
        """ Obtains a new batch.
        Args:
            randomize - not yet implemented
            do_not_fill - True/False
        """
        batch = []
        id_can_serve = self.can_serve()
        while id_can_serve is None:
            try:
                self.fill_batch_queues() #Fill up the queues
            except:
                print('File Pointer has reached the end')
                self.curr_epoch = self.curr_epoch+1
                if self.curr_epoch<self.epochs:
                    self.file_pointer = open(self.corp_path,'r',encoding='utf8')
                else:
                    print('Cannot serve more')
                    return None
            id_can_serve = self.can_serve()
            
        batch = [self.batch_queues[id_can_serve].pop() for _ in range(self.batch_size)]    
        return batch
    
class RawSentenceBatchGenerator(SentenceBatchGenerator):
    def __init__(self,corp_path,word2numb,**kwargs):
        kwargs['diff'] = kwargs.get('diff',kwargs.get('max_len',30)-kwargs.get('min_len',4))
        super().__init__(corp_path,word2numb,**kwargs)
        
    def fill_batch_queues(self,num_lines_read=100,randomize=True):
        for sentence in itertools.islice(self.file_pointer,num_lines_read):
            sentence = sentence.lower()[:-1]
            words = self._word_tokenizer.tokenize(sentence)
            if len(words)<self.min_len or len(words)>self.max_len:
                continue
            words_nums = self.word2numb.convert_w2n(words)
            if sum([w==self.UNK_ID for w in words_nums])/len(words) > self.unk_perc:
                continue
            idx =  bisect.bisect(self.queue_limits,len(words))-1
            if idx==self.numb_queues:
                idx = self.numb_queues-1
            if idx not in self.do_not_fill:
                self.batch_queues[idx].appendleft(sentence)
            if len(self.do_not_fill) == self.numb_queues:
                raise ValueError('no more queues can serve')
                
class RawSentenceBatchGeneratorLength(RawSentenceBatchGenerator):
    def fill_batch_queues(self,num_lines_read=100,randomize=True):
        for sentence in itertools.islice(self.file_pointer,num_lines_read):
            sentence = sentence.lower()[:-1]
            words = self._word_tokenizer.tokenize(sentence)
            if len(words)<self.min_len or len(words)>self.max_len:
                continue
            words_nums = self.word2numb.convert_w2n(words)
            if sum([w==self.UNK_ID for w in words_nums])/len(words) > self.unk_perc:
                continue
            idx =  bisect.bisect(self.queue_limits,len(words))-1
            if idx==self.numb_queues:
                idx = self.numb_queues-1
            if idx not in self.do_not_fill:
                self.batch_queues[idx].appendleft([sentence,len(words_nums)])  
            if len(self.do_not_fill) == self.numb_queues:
                raise ValueError('no more queues can serve')
                
def test_batch_gen():
    parent_dir = os.path.split(os.getcwd())[0]
    path_w2n_n2w = os.path.join(parent_dir, 'data', 'w2n_n2w_euro.pickle')
    path_to_eur = os.path.join(parent_dir, 'data', 'corpora', 'europarl-v7.en', 'europarl-v7.en')
    w2numb = Word2Numb(path_w2n_n2w)
    batch_gen = RawSentenceBatchGeneratorLength(path_to_eur,w2numb,diff=1)
    print(batch_gen.get_next_batch())
    
def test_w2n():
    dim=200
    parent_dir = os.path.split(os.getcwd())[0]
    path_corpus = os.path.join(parent_dir, 'data', 'corpora','news.2014.en.shuffled.v2' )
    path_w2n_n2w = os.path.join(parent_dir, 'data', 'w2n_n2w_news.pickle')
    path_word_embed = os.path.join(parent_dir, 'data', '%d_embed_news.pickle'%dim)
    w2n,n2w,e=word_dict_embed(80000,dim=200,path_corpus=path_corpus,path_w2n_n2w=path_w2n_n2w,path_word_embed=path_word_embed)

    
if __name__ == "__main__":
    pass
    parent_dir = os.path.split(os.getcwd())[0]
    path_w2n_n2w = os.path.join(parent_dir, 'data', 'w2n_n2w_euro.pickle')
    path_to_eur = os.path.join(parent_dir, 'data', 'corpora', 'europarl-v7.en', 'europarl-v7.en')
    w2numb = Word2Numb(path_w2n_n2w)
    batch_gen = RawSentenceBatchGeneratorLength(path_to_eur,w2numb,diff=1)
    print(batch_gen.get_next_batch())