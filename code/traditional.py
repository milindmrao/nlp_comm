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
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

def compress_batch(batch,**kwargs):
    """ Accepts a batch of sentences. Perhaps 32. Compresses it using a universal
    compressor such as gzip. Then expands it based on the parameters.
    Inputs:
        batch (list of sentences): list of sentences
        Optional params:
            'bit_drop_rate'(float or list-of-float): Default 0.01
            
        Returns:
            encoding (bit-stream)
            compression_ratio
            bits_per_sentence
    """
    n_sens = len(batch)
    sentences_all = str.encode(' '.join(batch))
    compressed = zlib.compress(sentences_all,9)
    
    bdr = int(2*np.ceil(255*kwargs.get('bit_drop_rate',0.01))) #This ensures that 0.01 * 255 (message size) gets reconstructed
    channel_coder = reedsolo.RSCodec(bdr)
    encoded = channel_coder.encode(compressed)
    
    comp_ratio = sys.getsizeof(encoded)/sys.getsizeof(sentences_all)
    bits_per_sentence = sys.getsizeof(encoded)/n_sens
                                     
    return (encoded,comp_ratio,bits_per_sentence)

if __name__=="__main__":
    parent_dir, _ = os.path.split(os.getcwd())    
    sentences_path = os.path.join(parent_dir, 'data', 'training_rest_reuters.pickle')
    
    batch_size = 32
    
    sentences = pickle.load(open(sentences_path,'rb'))

    #--------- Running the experiment to see what an effect batch size has -------------
    cpr_rat_mean=[]
    bps_mean = []
    batch_sizes = np.array(np.logspace(0,3,20),dtype=int)
    for batch_size in tqdm(batch_sizes):
        _,cpr_rat,bps = list(zip(*[compress_batch(sentences[x:x+batch_size]) 
            for x in range(0,len(sentences)-batch_size,batch_size)]))
        cpr_rat_mean.append(np.mean(cpr_rat))
        bps_mean.append(np.mean(bps)*8)
        
    plt.figure(1)
    plt.subplot(211)
    plt.plot(batch_sizes,bps_mean)
    plt.xlabel('Batch size')
    plt.ylabel('Bits per sentence')
    plt.ylim([300,800])
    plt.subplot(212)
    plt.plot(batch_sizes,cpr_rat_mean)
    plt.xlabel('Batch size')
    plt.ylabel('Compression ratio')
    plt.ylim([0.3,0.75])
    plt.savefig(os.path.join(parent_dir,'results','batch_bps_cpr'))
    
    #--------- Running the experiment about number of bits that the traditional method needs
    batch_size = 32
    cpr_rat_mean_bdr=[]
    bps_mean_bdr = []
    bit_drop_rates = np.linspace(0.01,0.2,10)
    for bdr in tqdm(bit_drop_rates):
        _,cpr_rat,bps = list(zip(*[compress_batch(sentences[x:x+batch_size],bit_drop_rate=bdr) 
            for x in range(0,len(sentences)-batch_size,batch_size)]))        
        cpr_rat_mean_bdr.append(np.mean(cpr_rat))
        bps_mean_bdr.append(np.mean(bps)*8)  
        
    plt.figure(1)
    plt.subplot(211)
    plt.plot(bit_drop_rates,bps_mean_bdr)
    plt.xlabel('Bit drop rate')
    plt.ylabel('Bits per sentence')
    plt.ylim([300,800])
    plt.subplot(212)
    plt.plot(bit_drop_rates,cpr_rat_mean_bdr)
    plt.xlabel('Bit drop rate')
    plt.ylabel('Compression ratio')
    plt.ylim([0.3,0.75])
    plt.savefig(os.path.join(parent_dir,'results','bdr_bps_cpr'))    