# -*- coding: utf-8 -*-
"""========================================================================
This script processes the test results produced by the neural network algorithm
and does some graphical processing on this.

@author: Nariman Farsad, and Milind Rao
@copyright: Copyright 2018
========================================================================"""


from editdistance import eval as edeval
import edit_distance
from preprocess_library import Word2Numb
import itertools
import bisect 
import numpy as np
from tqdm import tqdm
import os
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from jointSC import parse_args


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

def penn_to_wn(tag):
    """ converts tags from nltk to tags used by wordnet """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def get_synset(tagged):
    """ uses wordnet to get the synonyms for the input tagged word

        Args:
            tagged: a word that is tagged using nltk pos_tag

        Returns:
            syn: the synonyms set, i.e., synset for the word
    """
    lemmatzr = WordNetLemmatizer()
    wn_tag = penn_to_wn(tagged[0][1]) #convert tag to synset wordnet tag
    if not wn_tag: # if the tag not in synset wordnet tag, (e.g. for word our)
        return None

    # find the base of the word
    lemma = lemmatzr.lemmatize(tagged[0][0], pos=wn_tag)

    # find synset using synset wordnet tag
    syn = wn.synsets(lemma, pos=wn_tag)
    if not syn: # if not found
        syn = wn.synsets(lemma) # try without the synset wordnet tag
        if not syn:
            return None

    return syn[0] # return the first word in the synset


def edit_dist_with_repl_similarity(tx_numb,rx_numb,word2numb):
    """ This function aligns two seq according to edit distance and then
     subtracts the similarity measure between replaced words from the edit distance.
     Wu Parmer similarity measure is used for this task.

    Args:
        tx_numb: the number representation of the tx sentence
        rx_numb: the number representation fo the rx sentence
        word2numb: word to numb object

    Returns:
        dist_measur: returns the distance measure
    """
    # get the word representation
    tx_txt = word2numb.convert_n2w(tx_numb)
    rx_txt = word2numb.convert_n2w(rx_numb)

    ed_aligned = edit_distance.SequenceMatcher(a=tx_numb, b=rx_numb)

    dist_measur = ed_aligned.distance() # this is the edit distance

    indx_tx = 0
    indx_rx = 0
    # go through insertions and deletions and replacements in the alignment
    for i, op in enumerate(ed_aligned.get_opcodes()):
        # print(op)
        if op[0] == 'equal':
            indx_tx += 1
            indx_rx += 1
            continue
        elif op[0] == 'replace': # if replacement discount similarity
            tx_syn = get_synset(pos_tag([tx_txt[indx_tx]]))
            rx_syn = get_synset(pos_tag([rx_txt[indx_rx]]))
            sim = 0
            if (tx_syn is not None) and (rx_syn is not None):
                sim = tx_syn.wup_similarity(rx_syn) # use Wu Palmer similarity measure
                if sim is None:
                    sim = 0
            dist_measur -= sim
            indx_tx += 1
            indx_rx += 1
        elif op[0] == 'delete':
            indx_tx += 1
        elif op[0] == 'insert':
            indx_rx += 1
        else:
            print("****************** ERROR ***************")
            break

    return dist_measur

def calc_distance(tx_numb,rx_numb,word2numb, dist_type="ed_only"):
    """ This function returns the distance measure between two sequences
    depending on the type of the distance measure.

    Args:
        tx_numb: the number representation of the tx sentence
        rx_numb: the number representation fo the rx sentence
        word2numb: word to numb object
        dist_type: can be "ed_only" for edit distance or "ed_WuP" for
                   edit distance with discounted similarity measure.
                   Other types can be added in the future.

    Returns:
        dist_measur: returns the distance measure
    """
    if dist_type == "ed_only":
        return edeval(tx_numb,rx_numb)
    elif dist_type == "ed_WuP":
        return edit_dist_with_repl_similarity(tx_numb,rx_numb,word2numb)
    else:
        return None


def performance_test(test_path,word2numb,min_len=4,max_len=30,diff=2,
                     max_per_point = 50000,batch_size=1,dist_type="ed_WuP"):
    """ Function computes the performance index for the files using the edit
    distance or levenshtein metric for batches of different sentence lengths
    
    Args:
        test_path: path to the results tests path
        min_len: minimum length of the batch
        max_len: maximum length of the batch
        diff: difference
        max_per_point: maximum number of evaluations 
        batch_size: groups performance from batch_size evaluations before averaging
        dist_type: can be "ed_only" for edit distance or "ed_WuP" for
                   edit distance with discounted similarity measure.
                   Other types can be added in the future.
        
    Returns:
        performance: performance at each point
        perf_mean: mean 
        perf_std: standard deviation
        batch_limits: the limits of the batch
    """
    fop = open(test_path,'r',encoding='utf8')
    batch_limits = list(range(min_len,max_len,diff))

    performance = [[] for _ in range(len(batch_limits))]
    do_not_fill=set([])
    
    special_words = word2numb.convert_w2n(['<pad>','<end>','<start>'])
    
    for line in tqdm(fop,total=20000):
        tx_line = line[4:-7]
        
        try:
            rx_line = fop.readline()[4:-7]
            tx_words = simple_tokenizer(tx_line,word2numb,special_words)
            rx_words = remove_duplicates(simple_tokenizer(rx_line,word2numb,special_words))
        except:
            break
        
        idx =  bisect.bisect(batch_limits,len(tx_words))-1

        if idx not in do_not_fill:
            performance[idx].append(calc_distance(tx_words,rx_words,word2numb, dist_type)/len(tx_words))
        if len(performance[idx])>max_per_point*batch_size:
            do_not_fill.add(idx)
        if len(do_not_fill) == len(performance):
#            print('Filled all slots')
            break
    
    fop.close()

    performance = [batch_performance(row,batch_size) for row in performance]
    perf_mean = [np.mean(row) for row in performance]
    perf_std = [np.std(row) for row in performance]
    
    if len(performance)==1: #It is not divided by batches
        performance = performance[0]
        perf_mean = perf_mean[0]
        perf_std = perf_std[0]
        
    return performance, perf_mean, perf_std

def variation_exp(word2numb,bdr=0.95,bps=400,channel='erasure',var_bps_lin = 250,**kwargs):
    """ Returns performance of multiple files
    
    Args:
        word2numb - the dictionary object
        bdr - list or not of channel parameter values
        bps - number of bits per sentence
        channel
        var_bps_lin - lower limit in linear generation
    
    Returns:
        performance : size(bdr x bps)
        perf_mean
        perf_std
    """
    
    try:
        bps=list(bps)
    except:
        bps=[bps]
        var_bps_lin = [var_bps_lin]
    try:
        bdr = list(bdr)
    except:
        bdr = [bdr]
        
    performance,perf_mean,perf_std = [],[],[]  
    for bdr_curr,bps_curr in tqdm(itertools.product(bdr,bps),total=len(bdr)*len(bps),desc='bdr-bps/test '):
        
        var_bps_lin_curr = var_bps_lin[bps.index(bps_curr)]
        config_args = parse_args(['-c',channel,
                                  '-cp','{:0.2f}'.format(bdr_curr),
                                  '-ntx','{}'.format(bps_curr),
                                  '-d','news',
                                  '-t','beam',
                                  '-vv',
                                  '-bg','linear','{}'.format(var_bps_lin_curr)])
        test_path = config_args['test_results_path']
        perf_,perf_mean_,perf_std_ = performance_test(test_path,word2numb,diff = 30,**kwargs)
        performance.append(perf_)
        perf_mean.append(perf_mean_)
        perf_std.append(perf_std_)
    
    return performance,perf_mean,perf_std
                
    
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
    pass
#    parent_dir = os.path.split(os.getcwd())[0]
#    path_w2n_n2w = os.path.join(parent_dir, 'data', 'w2n_n2w_euro.pickle')
#    path_to_eur = os.path.join(parent_dir, 'data', 'corpora', 'europarl-v7.en', 'europarl-v7.en')
#    w2numb = Word2Numb(path_w2n_n2w)
#    test_path = os.path.join(parent_dir,'results','test_results',
#                             'Chan-erasure0.95-lr-0.001-txbits-400-voc-19158-embed-200-lstm-256-peep-True-epochs-30-bs-1-SOSEuroSentence-Binarizer-HardAttn-0EncLayers-2-DecLayers-2-VarProbCorrect-MaxSentLen-30-BeamSearch-10.txt')
#    print('evaluating from test')
##    perf_nn_all,perf_nn_mean,perf_nn_std,b_lim = performance_test(test_path,w2numb,diff=10,max_per_point=50)
#    perf,perf_mean,perf_std = variation_test(w2numb,bdr=[0,0.05],max_per_point=20)
    