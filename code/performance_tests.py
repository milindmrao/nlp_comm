# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:43:25 2017

This script processes the test results produced by the neural network algorithm
and does some graphical processing on this. 

@author: Milind
"""

from editdistance import eval as edeval
from preprocess_library import Word2Numb
import itertools
import bisect 
import numpy as np
from tqdm import tqdm
import os


def simple_tokenizer(sentence,word2numb,special_words=[]):
    """ Simple tokenization based on space and removal of special characters"""
    words = word2numb.convert_w2n(sentence.split(' '))
    words = list(filter(lambda x: x not in special_words,words))
    return words

def remove_duplicates(words_nums):
    """ removes contiguous duplicates"""
    words_nums_no_rep,_ = zip(*itertools.groupby(words_nums))
    return words_nums_no_rep

def batch_performance(row,batch_size=32):
    """ Groups contiguous groups of values of size batch_size"""
    row_batches = zip(*([iter(row)]*batch_size))
    return [np.mean(x) for x in row_batches]

def performance_test(test_path,word2numb,min_len=4,max_len=30,diff=2,max_per_point = 50,batch_size=32):
    """ Function computes the performance index for the files using the edit
    distance or levenshtein metric
    
    Args:
        test_path: path to the results tests path
        min_len: minimum length of the batch
        max_len: maximum length of the batch
        diff: difference
        max_per_point: maximum number of evaluations 
        batch_size: groups performance from batch_size evaluations before averaging
        
    Returns:
        performance: performance at each point
        perf_mean: mean 
        perf_std: standard deviation
        batch_limits: the limits of the batch
    """
    fop = open(test_path,'r',encoding='utf8')
    batch_limits = list(range(min_len,max_len,diff)) + [max_len]

    performance = [[] for _ in range(len(batch_limits)-1)]
    do_not_fill=set([])
    
    special_words = word2numb.convert_w2n(['<pad>','<end>','<start>'])
    
    for line in tqdm(fop,total=200000):
        tx_line = line[4:-1]
        rx_line = fop.readline()[4:-1]
        
        try:
            tx_words = simple_tokenizer(tx_line,word2numb,special_words)
            rx_words = remove_duplicates(simple_tokenizer(rx_line,word2numb,special_words))
        except:
            break
        
        idx =  bisect.bisect(batch_limits,len(tx_words))-1
        if idx==len(performance):
            idx = len(performance)-1
        if idx not in do_not_fill:
            performance[idx].append(edeval(tx_words,rx_words)/len(tx_words))  
        if len(performance[idx])>max_per_point*batch_size:
            do_not_fill.add(idx)
        if len(do_not_fill) == len(performance):
            print('Filled all slots')
            break
    
    fop.close()

    performance = [batch_performance(row,batch_size) for row in performance]
    perf_mean = [np.mean(row) for row in performance]
    perf_std = [np.std(row) for row in performance]
    return performance, perf_mean, perf_std

def variation_test(word2numb,bdr=0.05,bps=400,max_per_point = 50,batch_size=32):
    """ Tests multiple files at various points
    Args:
        word2numb: a word2number object
        bdr: the bit drop rate. Could be a list
        bps: bits per sentence
        max_per_point: maximum number of batches of sentences
        batch_size: size of each batch
        
    Returns:
        performance : size(bdr x bps)
        perf_mean
        perf_std
    """
    
    parent_dir = os.path.split(os.getcwd())[0]
    
    try:
        bps=list(bps)
    except:
        bps=[bps]
    try:
        bdr = list(bdr)
    except:
        bdr=[bdr]
    
    performance,perf_mean,perf_std = [],[],[]  
    special_words = word2numb.convert_w2n(['<pad>','<end>','<start>'])
    for bdr_curr,bps_curr in tqdm(itertools.product(bdr,bps),total=len(bdr)*len(bps),desc='bdr-bps/test '):
        test_path = os.path.join(parent_dir,'results','test_results',
                                 'd{:.2f}-b{:3d}.txt'.format(bdr_curr,bps_curr))
        if not os.path.exists(test_path):
            print('{} does not exist'.format(test_path))
            continue
        fop = open(test_path,'r',encoding='utf8')
        perf_point = []
        for line in tqdm(fop,total=200000,desc='test_file '):
            if np.random.rand()>(max_per_point*batch_size)/200000: #scrambling
                fop.readline()
                continue
            
            tx_line = line[4:-1]
            rx_line = fop.readline()[4:-1]
            
            try:
                tx_words = simple_tokenizer(tx_line,word2numb,special_words)
                rx_words = remove_duplicates(simple_tokenizer(rx_line,word2numb,special_words))
            except:
                break
            
            perf_point.append(edeval(tx_words,rx_words)/len(tx_words))
        
        fop.close()
        np.random.shuffle(perf_point)
        perf_point = batch_performance(perf_point,batch_size)
        perf_mean_point = np.mean(perf_point) 
        perf_std_point = np.std(perf_point)
        
        performance.append(perf_point)
        perf_mean.append(perf_mean_point)
        perf_std.append(perf_std_point)
        
    return performance,perf_mean,perf_std
    
if __name__=="__main__":
    parent_dir = os.path.split(os.getcwd())[0]
    path_w2n_n2w = os.path.join(parent_dir, 'data', 'w2n_n2w_euro.pickle')
    path_to_eur = os.path.join(parent_dir, 'data', 'corpora', 'europarl-v7.en', 'europarl-v7.en')
    w2numb = Word2Numb(path_w2n_n2w)
    test_path = os.path.join(parent_dir,'results','test_results',
                             'Chan-erasure0.95-lr-0.001-txbits-400-voc-19158-embed-200-lstm-256-peep-True-epochs-30-bs-1-SOSEuroSentence-Binarizer-HardAttn-0EncLayers-2-DecLayers-2-VarProbCorrect-MaxSentLen-30-BeamSearch-10.txt')
    print('evaluating from test')
#    perf_nn_all,perf_nn_mean,perf_nn_std,b_lim = performance_test(test_path,w2numb,diff=10,max_per_point=50)
    perf,perf_mean,perf_std = variation_test(w2numb,bdr=[0,0.05],max_per_point=20)
    