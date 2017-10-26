# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:01:21 2017

@author: Milind
a competitive method - running a universal compressor 
"""

import numpy as np
import zlib
import reedsolo 
import sys
import os
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import collections
from huffman import codebook as hcode
from functools import lru_cache
from preprocess_library import RawSentenceBatchGeneratorLength, Word2Numb
import bisect
from performance_tests import performance_test, variation_test

class traditional(object):
    def source(self,sentences):
        """ gzips the sentences
        Args:
            sentences: any string
        Returns:
            gzipped object
        """
        return zlib.compress(str.encode(sentences),9)
    
    @lru_cache(maxsize=32) 
    def rs_codec(self,bdr_byte):
        """ Function returns an RS Codec for fixed bit drop rate. Results are cached
        """
        return reedsolo.RSCodec(bdr_byte)
        
    def channel(self,obj_to_chan_code,bdr=0.05):
        """ Does reed-solomon coding
        Args:
            obj_to_chan_code 
            bdr: bit drop rate
        Returns:
            reed-solomon code
        """
        bdr_byte = int(np.ceil(255*bdr))
        channel_coder = self.rs_codec(bdr_byte)
        return channel_coder.encode(obj_to_chan_code)

    def source_channel(self,sentences,**kwargs):
        """ Accepts a sentence. gzips it and uses reed-solomon. Returns the 
        encoded object
        Args:
            sentences - string
            bdr - bit drop rate
        """
        bdr = kwargs.get('bdr',0.05)
        return self.channel(self.source(sentences),bdr)
    
    def compress_batch(self,batch,**kwargs):
        """ Accepts a batch of sentences. Perhaps 32. Compresses it using a universal
        compressor such as gzip. Then expands it based on the parameters.
        Args:
            batch (list of sentences): list of sentences
            Optional params:
                bdr(float or list-of-float): bit_drop_rate. Default 0.01
                
        Returns:
            encoding (bit-stream)
            compression_ratio
            bits_per_sentence
        """
        n_sens = len(batch)
        sentences_all = ' '.join(batch)
        encoded = self.source_channel(sentences_all,**kwargs)
        
        comp_ratio = sys.getsizeof(encoded)/sys.getsizeof(sentences_all)
        bits_per_sentence = sys.getsizeof(encoded)/n_sens*8
                                         
        return (encoded,comp_ratio,bits_per_sentence)

    def performance_batch(self,batch,batch_lens,bdr=0.05,bps=500):
        """ Computes the performance of the traditional method given bit drop rate
        and bits per sentence restrictions
        Args:
            batch - batch of sentences
            batch_lens - lengths of the sentences
            bdr - bit drop rate
            bps - bits per sentence
        Returns:
            performance - word error rate accurate to second decimal place
        """
        batch_size = len(batch)
        low_index = 0
        high_index = batch_size-1
        while high_index>=low_index:
            index_to_check = int((low_index+high_index)/2)
            encoding = self.source_channel(' '.join(batch[:index_to_check+1]),bdr=bdr)
            bits_per_sentence = self.getsize(encoding)/batch_size
            if bits_per_sentence> 1.01*bps:
                high_index = index_to_check - 1
            elif bits_per_sentence< 0.99*bps:
                low_index = index_to_check + 1
            else:
                break
        word_error_rate = 1-sum(itertools.islice(batch_lens,index_to_check+1))/sum(batch_lens)
        return word_error_rate
    
    def performance_batches(self,batches,bdr=0.05,bps=500,verbose=False):
        """ Wrapper around performance_batch that computes the performance for 
        multiple batches
        Args:
            batches : a list of batches. Each has sentence,sentence_len
            bdr
            bps
            verbose: whether to print progress
        returns:
            list of performance values
        """
        if verbose:
            batch_iter = tqdm(batches,desc=self.__class__.__name__)
        else:
            batch_iter = batches
            
        return [self.performance_batch(bat,bat_lens,bdr,bps) for bat,bat_lens 
                in map(lambda x:zip(*x),batch_iter)]
    
    def bit_string_to_byte(self,bit_string):
        return int(bit_string, 2).to_bytes((len(bit_string) + 7) // 8, 'big')
    
    def getsize(self,encoding):
        """ Returns the size of an encoding. Works for string or byte sequence
        
        Args:
            encoding: binary string or byte sequence
        Returns:
            integer
        """
        try:
            int(encoding,2) #String is binary
            return len(encoding)
        except ValueError: #String is byte sequence
            return len(encoding)*8 
            
class bit5_rs(traditional):
    def __init__(self):
        self.chars = string.ascii_lowercase + " ,.?!'"
        self.char_bin = dict((char, '{0:05b}'.format(index)) for index,char in enumerate(self.chars))
        self.bin_char = dict((value,key) for key,value in self.char_bin.items())
        
    def source(self,sentences):
        encoding = ''.join(self.char_bin.get(x,self.char_bin['x']) for x in sentences)
        return encoding
    
    def channel(self,encoding,bdr=0.05):
        """ Does channel coding using turbo/ldpc/convolutional codes
        """
        obj_to_chan_code = self.bit_string_to_byte(encoding)
        return super().channel(obj_to_chan_code,bdr)
    
    
class huffman_rs(traditional):
    def __init__(self,path_corpus,num_lines_read=50000):
        fop = open(path_corpus,'r',encoding='utf8')
        char_count = collections.Counter()
        print('initializing huffman codebook')
        for line in tqdm(itertools.islice(fop,num_lines_read),total=num_lines_read):
            char_count.update(line.lower()[:-1])
        fop.close()
        self.huffman_code = hcode(char_count.items())
        
    def source(self,sentences):
        """ Returns binary string sequence"""
        encoding = ''.join(self.huffman_code.get(x,self.huffman_code['?']) for x in sentences)
        return encoding
    
    def channel(self,encoding,bdr=0.05):
        obj_to_chan_code = self.bit_string_to_byte(encoding)
        return super().channel(obj_to_chan_code,bdr=bdr)

def mean_std_performance(performance):
    """ takes in a performance object which is a dictionary whose keys 
    are the various models tried out. The values are lists of lists 
    (or a matrix in a rare case). It computes the means and variances
    across the rows of this matrix
    """
    performance_mean = dict((name,[np.mean(row) for row in values]) 
                        for name,values in performance.items())
    performance_std = dict((name,[np.std(row) for row in values])
                        for name,values in performance.items())
    return performance_mean,performance_std

def variation(word2numb,bdr=0.05,bps = list(range(300,700,50)),max_per_point=50,batch_size=32):
    """ Function loops through bdr, bps values and produced word error 
    rates for all the models considered.
    
    Args:
        word2numb
        bdr
        bps
        max_per_point
        batch_size
        
    Returns:
        performance - dictionary of 'trad', 'huffman', 'bit5'
        perf_mean
        perf_std
    """
    batch_gen = RawSentenceBatchGeneratorLength(path_to_eur,word2numb,batch_size=batch_size,diff=50)
    try:
        bdr = list(bdr)
    except:
        bdr = [bdr]
    
    try:
        bps=list(bps)
    except:
        bps=[bps]
        
    n2model = {}
    n2model['trad'] = traditional()
    n2model['bit5'] = bit5_rs()
    n2model['huffman'] = huffman_rs(path_to_eur)
    performance = {'trad':[],'bit5':[],'huffman':[]}
    
    for bdr_iter,bps_iter in tqdm(itertools.product(bdr,bps),total = len(bdr)*len(bps),desc='bdr-bps '):
        batches = [batch_gen.get_next_batch() for _ in range(max_per_point)]
        [performance[name].append(
            model.performance_batches(batches,bdr_iter,bps_iter)) 
            for name,model in n2model.items()]
        
    perf_mean,perf_std = mean_std_performance(performance)
    return performance,perf_mean, perf_std

def variation_sentence_length(word2numb,bdr=0.05,bps=400,max_per_point = 50,diff=2,batch_size=50):
    """ Function loops through batches of different lengths. It then produces 
    a word error rate for different sentence lengths. This is for a fixed bdr
    and bps
    Args:
        word2numb
        bdr - the bit drop rate
        bps - number of bits per sentence
        max_per_point - the maximum number of batches to evaluate of a 
                        particular sentence length
        batch_size
    Returns:
        performance - dictionary of 'trad', 'huffman', 'bit5'
        perf_mean
        perf_std
    """
    batch_gen = RawSentenceBatchGeneratorLength(path_to_eur,word2numb,batch_size=batch_size,diff=diff)
    n2model = {}
    n2model['trad'] = traditional()
    n2model['bit5'] = bit5_rs()
    n2model['huffman'] = huffman_rs(path_to_eur)
    batch_limits = batch_gen.queue_limits.copy()
    performance = dict((key,[[] for _ in range(batch_gen.numb_queues)]) for key in n2model)
    
    for ind in tqdm(range(max_per_point)):
        
        batch_batch_lens = batch_gen.get_next_batch()
        if not batch_batch_lens:
            print('Finished serving batches')
            break
        else:
            batch,batch_lens = zip(*batch_batch_lens)
            
        mean_sentence_len = np.mean(batch_lens)    
        idx =  bisect.bisect(batch_limits,mean_sentence_len)-1
        if idx==batch_gen.numb_queues:
            idx = batch_gen.numb_queues-1
        if len(performance['trad'][idx])> max_per_point:
            batch_gen.update_do_not_fill(idx)
            continue  #We have too many batches of a particular length
        [performance[key][idx].append(
                model.performance_batch(batch,batch_lens,bdr,bps)) 
                for key,model in n2model.items()]
        
    perf_mean,perf_std = mean_std_performance(performance)
    return performance, perf_mean, perf_std


    
if __name__=="__main__":
    parent_dir = os.path.split(os.getcwd())[0]
    path_w2n_n2w = os.path.join(parent_dir, 'data', 'w2n_n2w_euro.pickle')
    path_to_eur = os.path.join(parent_dir, 'data', 'corpora', 'europarl-v7.en', 'europarl-v7.en')
    w2numb = Word2Numb(path_w2n_n2w)
    batch_size=32
    max_per_point = 50 
    
#   #---------- Word error rate as number of bits changes ---------------------
       
#    bps = [350,400,450,500,550,600]
#    bdr = 0.05
#    perf_all,perf_mean,perf_std = variation(w2numb,bdr,bps,max_per_point=max_per_point,batch_size=batch_size)
#    perf_nn_all,perf_nn_mean,perf_nn_std = variation_test(w2numb,bdr=bdr,bps=bps,max_per_point = max_per_point,batch_size=batch_size)
#    plt.figure(1)
#    plt.errorbar(bps,perf_mean['bit5'],yerr=perf_std['bit5'],fmt='k:',capsize=2)
#    plt.errorbar(bps,perf_mean['huffman'],yerr=perf_std['huffman'],fmt='r--',capsize=2)
#    plt.errorbar(bps,perf_mean['trad'],yerr=perf_std['trad'],capsize=2)
#    plt.errorbar(bps,perf_nn_mean,yerr = perf_nn_std,fmt='g-.',capsize=2)
#    plt.xlabel('Bits per sentence')
#    plt.ylabel('Word error rate')
#    plt.legend(['bit5','huffman','gzip','DeepNN'])
#    plt.savefig(os.path.join(parent_dir,'results','bps_word_error.eps'))
 


#   #---------- Word error rate as the bdr changes --------------------------   
    bps = 400
    bdr = [0,0.05,0.1,0.15,0.2]
    perf_all,perf_mean,perf_std = variation(w2numb,bdr,bps,max_per_point=max_per_point,batch_size=batch_size)
    perf_nn_all,perf_nn_mean,perf_nn_std = variation_test(w2numb,bdr=bdr,bps=bps,max_per_point = max_per_point,batch_size=batch_size)
    plt.figure(2)
    plt.errorbar(bdr,perf_mean['bit5'],yerr=perf_std['bit5'],fmt='k:',capsize=2)
    plt.errorbar(bdr,perf_mean['huffman'],yerr=perf_std['huffman'],fmt='r--',capsize=2)
    plt.errorbar(bdr,perf_mean['trad'],yerr=perf_std['trad'],capsize=2)
    plt.errorbar(bdr,perf_nn_mean,yerr = perf_nn_std,fmt='g-.',capsize=2)
    plt.xlabel('Bit drop rate')
    plt.ylabel('Word error rate')
    plt.legend(['bit5','huffman','gzip','DeepNN'])
    plt.savefig(os.path.join(parent_dir,'results','bdr_word_error.eps'))



#   #-------------- Performance of sentence with sentence length -------------------
#    diff=1
#    max_per_point = 200
#    batch_gen = RawSentenceBatchGeneratorLength(path_to_eur,w2numb,batch_size=32,diff=diff)
#    perf_len_all,perf_len_mean,perf_len_std = variation_sentence_length(w2numb,max_per_point=max_per_point,diff=diff,batch_size=batch_size)
#    lens_sentences = [(x+y)/2 for x,y in zip(batch_gen.queue_limits[:-1],batch_gen.queue_limits[1:])]
#    test_path = os.path.join(parent_dir,'results','test_results','d0.05-b400.txt')
#    print('evaluating from test')
#    perf_nn_all,perf_nn_mean,perf_nn_std = performance_test(test_path,w2numb,diff=diff,max_per_point=max_per_point,batch_size=batch_size)
#    plt.figure(3)
#    plt.errorbar(lens_sentences,perf_len_mean['bit5'],yerr=perf_len_std['bit5'],fmt='k:',capsize=2)
#    plt.errorbar(lens_sentences,perf_len_mean['huffman'],yerr=perf_len_std['huffman'],fmt='r--',capsize=2)
#    plt.errorbar(lens_sentences,perf_len_mean['trad'],yerr=perf_len_std['trad'],capsize=2)
#    plt.errorbar(lens_sentences,perf_nn_mean,yerr=perf_nn_std,fmt='g-.',capsize=2)
#    plt.xlabel('Sentence length')
#    plt.ylabel('Word error rate')
#    plt.legend(['bit5','huffman','gzip','DeepNN'])
#    plt.savefig(os.path.join(parent_dir,'results','len_sentence_word_error.eps'))
    
    
#    #--------- Legacy code unused ------------------------    
#    parent_dir, _ = os.path.split(os.getcwd())    
#    sentences_path = os.path.join(parent_dir, 'data', 'training_rest_reuters.pickle')
#    
#    batch_size = 32
#    
#    sentences = pickle.load(open(sentences_path,'rb'))
#
#    #--------- Running the experiment to see what an effect batch size has -------------
#    cpr_rat_mean=[]
#    bps_mean = []
#    batch_sizes = np.array(np.logspace(0,3,20),dtype=int)
#    for batch_size in tqdm(batch_sizes):
#        _,cpr_rat,bps = list(zip(*[compress_batch(sentences[x:x+batch_size]) 
#            for x in range(0,len(sentences)-batch_size,batch_size)]))
#        cpr_rat_mean.append(np.mean(cpr_rat))
#        bps_mean.append(np.mean(bps)*8)
#        
#    plt.figure(1)
#    plt.subplot(211)
#    plt.plot(batch_sizes,bps_mean)
#    plt.xlabel('Batch size')
#    plt.ylabel('Bits per sentence')
#    plt.ylim([300,800])
#    plt.subplot(212)
#    plt.plot(batch_sizes,cpr_rat_mean)
#    plt.xlabel('Batch size')
#    plt.ylabel('Compression ratio')
#    plt.ylim([0.3,0.75])
#    plt.savefig(os.path.join(parent_dir,'results','batch_bps_cpr'))
#    
#    #--------- Running the experiment about number of bits that the traditional method needs
#    batch_size = 32
#    cpr_rat_mean_bdr=[]
#    bps_mean_bdr = []
#    bit_drop_rates = np.linspace(0.01,0.2,10)
#    for bdr in tqdm(bit_drop_rates):
#        _,cpr_rat,bps = list(zip(*[compress_batch(sentences[x:x+batch_size],bit_drop_rate=bdr) 
#            for x in range(0,len(sentences)-batch_size,batch_size)]))        
#        cpr_rat_mean_bdr.append(np.mean(cpr_rat))
#        bps_mean_bdr.append(np.mean(bps)*8)  
#        
#    plt.figure(1)
#    plt.subplot(211)
#    plt.plot(bit_drop_rates,bps_mean_bdr)
#    plt.xlabel('Bit drop rate')
#    plt.ylabel('Bits per sentence')
#    plt.ylim([300,800])
#    plt.subplot(212)
#    plt.plot(bit_drop_rates,cpr_rat_mean_bdr)
#    plt.xlabel('Bit drop rate')
#    plt.ylabel('Compression ratio')
#    plt.ylim([0.3,0.75])
#    plt.savefig(os.path.join(parent_dir,'results','bdr_bps_cpr'))    