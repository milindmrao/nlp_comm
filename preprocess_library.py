# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:43:13 2017

This is the library of preprocessing functions

@author: Milind, Nariman
"""

import os
import numpy as np
from tqdm import tqdm
import nltk
import pickle
from collections import deque
from bisect import bisect
import itertools
import argparse
import logging 

# Pre-initializing what the constants should be
PAD_ID = 0
END_ID = 1
START_ID = 2
UNK_ID = 3

def word_dict_embed(vocab_size=100000, **kwargs):
    """This function creates the word2num, num2word and embeddings objects. It 
    first goes through the corpus to extract word counts. It then uses these and 
    a pretrained embeddings file to 
    
    Args:
        vocab_size: int. default 50000
        dim: int. Dimensionality of word embeddings. Default 200
        path_corpus: path of the text corpora
        path_embed: path of the embeddings file. Default is constructed from dim
           size
        path_w2n_n2w: path to save the w2n_n2w file
        path_word_freq: path to save the word frequency file
        path_word_embed: path to save the final word embeddings
        iter_limit: maximum number of lines to read in the corpus file.
    Returns:
        
    """
    parent_dir = os.path.split(os.getcwd())[0]
    dim = kwargs.get('dim',200)
    dataset = kwargs.get('dataset','giga')
    path_to_datacorp = os.path.join(parent_dir, 'data', dataset, 
                                    '{}_train.dat'.format(dataset))
    path_corpus = kwargs.get('path_corpus',path_to_datacorp)
    path_glove_embed = os.path.join(parent_dir,'data',
                                    'glove.6b.{}d.txt'.format(dim))
    path_embed = kwargs.get('path_embed',path_glove_embed)
    path_w2n_n2w = kwargs.get('path_w2n_n2w',
                              os.path.join(parent_dir, 'data', dataset,
                                          'w2n_n2w_{}.pickle'.format(dataset)))
    path_word_freq = kwargs.get('path_word_freq', 
                              os.path.join(parent_dir, 'data', dataset,
                                        'word_freq_{}.pickle'.format(dataset)))
    path_word_embed = kwargs.get('path_word_embed',
                                 os.path.join(parent_dir, 'data', dataset,
                                     '{}_embed_{}.pickle'.format(dim,dataset)))
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
    
    #-------- using word frequencies to create word2num, num2word, word_freq -------
    words_special = [('<pad>', PAD_ID), ('<end>', END_ID), ('<start>', START_ID), ('<unk>', UNK_ID)]
    len_words_special = len(words_special)
    
    common_words, common_word_freq = zip(*word_freq.most_common(vocab_size))
    common_words_index = zip(common_words,range(len_words_special,len_words_special + len(common_words)))
    words_index = words_special + list(common_words_index)
    
    word2num = dict(words_index)
    num2word = dict([n,w] for (w,n) in word2num.items())
    
    with open(path_w2n_n2w, 'wb') as fop:
        pickle.dump([word2num, num2word], fop)
    
    common_word_freq = [0 for _ in range(len_words_special)] + list(common_word_freq)
    common_word_freq[UNK_ID] = sum(word_freq.values()) - sum(common_word_freq)
    with open(path_word_freq, 'wb') as fop:
        pickle.dump(common_word_freq, fop)
        
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
        
    return word2num,num2word,embedding,common_word_freq


class Word2Numb(object):
    def __init__(self, w2n_path,vocab_size=None):
        # ==== load num2word and word2num =======
        
        with open(w2n_path, 'rb') as fop:
            [w2n, n2w] = pickle.load(fop)
        if vocab_size and vocab_size<len(n2w):
            self.n2w = dict((num,n2w[num]) for num in range(vocab_size))
            self.w2n = dict((word,num) for num,word in self.n2w.items())
        else:
            self.n2w, self.w2n = n2w,w2n
        self.vocab_size = len(self.n2w)
        logging.info('loaded dictionary of size {}'.format(self.vocab_size))
        self.UNK_ID = self.w2n['<unk>']

    def convert_w2n(self, sentence):
        return [self.w2n.get(x, self.UNK_ID) for x in sentence]

    def convert_n2w(self, numbs):
        return [self.n2w.get(x, "<>") for x in numbs]

class BatchGenerator (object):
    """ Class that does some processing on input files, batches them based on 
    length and can fill up the input pipeline 
    """
    def __init__(self, 
                 corp_path,
                 word2numb,
                 mode='sent_std',
                 **kwargs):
        """ 
        Args:
            corp_path: path to the corpus - could be file or folder. 
                Currently assume each line is a new sentence
            word2numb: word2numb object 
            mode: 'sent_std' dataset of sentences, returns tokens in sentence
                  'sent_rawl' dataset of sentences, returns eligible sentence in
                      raw form (lower-cased, unk-conversion) with length
                  'summ_std' dataset of sentence, next line summary. Returns 
                      tokens in both
                  'summ_rawl' dataset of sent, summ. Returns raw sentence, 
                      length
            batch_size: size of each batch (32)
            min_len: minimum length of sentence in the queue (4)
            max_len: maximum length of sentence in the queue (30)
            diff: the max difference between sentence length in each batch (4)
            epochs: the number of epochs. def: 10
            unk_perc: the unknown word percentage. default: 0.2
            vocab_out: vocabulary in output. Default word2numb vocab
        """

        self.corp_path = corp_path
        self.word2numb = word2numb
        self.batch_size = kwargs.get('batch_size',32)
        self.UNK_ID = word2numb.w2n['<unk>']
        self.unk_perc = kwargs.get('unk_perc',0.2)
        self.epochs = kwargs.get('epochs',10)
        self.curr_epoch = 0
        self.mode = mode
        self.vocab_out = kwargs.get('vocab_out',self.word2numb.vocab_size)
        self.min_len = kwargs.get('min_len',4)
        self.max_len = kwargs.get('max_len',30)
        diff = kwargs.get('diff',4)
        self.init_batch_queues(self.min_len,self.max_len,diff)
        self._word_tokenizer = nltk.tokenize.WordPunctTokenizer()

    def init_batch_queues(self,min_len,max_len,diff):
        self.batch_queues = []
        self.queue_limits = list(range(min_len,max_len,diff)) + [max_len+1]
        self.numb_queues = len(self.queue_limits)-1
        self.batch_queues = [deque() for _ in range(self.numb_queues)]
                
    def fill_sent_in_queue(self,len_sentence,entity_to_fill):
        """ Fills a sentence or entity in the queue in the appropriate slot"""
        idx =  bisect(self.queue_limits,len_sentence)-1
        self.batch_queues[idx].appendleft(entity_to_fill)
    
    def len_test(self,word_list):
        """ True if word_list is of valid length """ 
        return self.min_len<=len(word_list)<=self.max_len
    
    def unk_test(self,word_num_list):
        """ True if number of unknowns is not too high """
        return word_num_list.count(self.UNK_ID)/len(word_num_list)<=self.unk_perc
    
    def fill_batch_queues(self, file_pointer,num_lines_read=100):
        """ This function reads in num_lines_read number of lines. 
        It then converts it numbers, does filtering and places it in the appropriate batch
        
        Args:
            file_pointer: file_pointer to the file that is being read
            num_lines_read: number of lines to read at once
        """
        sentence = None
#        if self.mode in ['summ_std','summ_rawl']:
#            num_lines_read = 2*num_lines_read
        for sentence in itertools.islice(file_pointer,num_lines_read):
            #Reading sentence the first line
            sentence = sentence.lower()[:-1]
            words = self._word_tokenizer.tokenize(sentence)
            words_nums = self.word2numb.convert_w2n(words)
            is_ineligible = not self.len_test(words) or not self.unk_test(words_nums)
            
            if self.mode in ['summ_std','summ_rawl']:
                summ = file_pointer.readline().lower()[:-1]
                summ_words = self._word_tokenizer.tokenize(summ)
                summ_words_nums = list(map(lambda x: x if x<self.vocab_out else
                                       self.UNK_ID,
                                       self.word2numb.convert_w2n(summ_words)))
                is_ineligible = is_ineligible or not self.unk_test(summ_words_nums) \
                                        or not self.len_test(summ_words_nums)
                entity_to_fill = (words_nums,summ_words_nums) if \
                    self.mode=='summ_std' else (sentence,len(words))
            else:
                entity_to_fill = (words_nums, words_nums) if self.mode=='sent_std' \
                    else (sentence, len(words))
                    
            if is_ineligible: continue
            self.fill_sent_in_queue(len(words), entity_to_fill)
        if sentence is None:
            raise ValueError('file pointer has reached the end')
            
        
    def can_serve(self, randomize=True):
        """ Returns the id of the queue which is in a position to serve"""
        idxes = list(filter(
                lambda x: len(self.batch_queues[x])>=self.batch_size,
                              range(self.numb_queues)))
        if idxes:
            if randomize:
                rand_id = np.random.randint(0,high=len(idxes))
                return idxes[rand_id]
            else:
                return idxes[0]
        else:
            return None

    def get_next_batch(self,randomize=True,**kwargs):
        """ Obtains a new batch.
        Args:
            randomize - whether to draw any available batch or the shortest
                one
        """
        logging.info('Reading file {} time'.format(self.curr_epoch))
        self.curr_epoch += 1
        with open(self.corp_path,'r', encoding='utf8') as fop:
            while True:
                batch = []
                id_can_serve = self.can_serve(randomize)
                while id_can_serve is None:
                    try:
                        self.fill_batch_queues(fop)
                    except ValueError:
                        break
                    id_can_serve = self.can_serve()
                else:    
                    # Executed only if id_can_serve 
                    batch = [self.batch_queues[id_can_serve].pop() for 
                                 _ in range(self.batch_size)]    
                    yield batch
                    continue
                break
    

def line_count_stats(pathname,limit=int(1e6),length_from=4,length_to=30,bin_len=4,*kwargs):
    """ Returns the relative frequency of the batches of different lengths for 
    computation of number of bits to allow per batch
    """
    freq = {}
    word_token = nltk.tokenize.WordPunctTokenizer()
    with open(pathname,'r',encoding='utf8') as fop:
        for line in tqdm(itertools.islice(fop,limit)):
            num_words = len(word_token.tokenize(line))
            freq[num_words] = freq.get(num_words,0)+1
    
    freq_rel = [freq[_] for _ in range(4,31)]
    freq_rel_bat = [sum(x) for x in itertools.zip_longest(*([iter(freq_rel)]*4),fillvalue = 0)]
    freq_rel_bat_norm = [x/sum(freq_rel_bat) for x in freq_rel_bat]
    return freq_rel_bat_norm

def create_line_stats():
    """ Creates and stores line stats
    """
    freq= {}
    freq['wiki'] = line_count_stats('../data/wiki/wiki_test.dat')
    freq['euro'] = line_count_stats('../data/corpora/europarl-v7.en/europarl-v7.en')
    freq['news'] = line_count_stats('../data/corpora/news/news.2016.en.shuffled')
    pickle.dump(freq,open('../data/corp_freq.pickle','wb'))
    
def bin_batch_create(numb_tx_bits,dataset='euro',type_bit_bin = 'const',low_lim=None,**kwargs):
    """ Creates bits per bin for std 4-4-30 line length config from dataset
    Args:
        numb_tx_bits - number of transmission bits
        dataset - which dataset. Can be euro, wiki, or news
        type_bit_bin - can be const, linear, or sqrt
        low_lim - how many bits for the small sentences batch
        
    Returns:
        bits per bin
    """
    if low_lim == None:
        low_lim = round(0.6*numb_tx_bits)
    freq = pickle.load(open('../data/corp_freq.pickle','rb'))
    len_bats = len(freq[dataset])
    if type_bit_bin == 'const':
        return [numb_tx_bits]*len_bats
    elif type_bit_bin == 'linear':
        func_i = [i for i in range(len_bats)]
    elif type_bit_bin == 'sqrt':
        func_i = [np.sqrt(i) for i in range(len_bats)]
    
    step = (numb_tx_bits-low_lim)/sum(freq[dataset][i]*func_i[i] for i in range(len_bats))
    bits_per_bin = [round(low_lim + step*func_i[i]) for i in range(len_bats)]
    return bits_per_bin


def generate_tb_filename(conf_args):
    """Generate the file name for the model
    """
    tb_name = conf_args['dataset']+'-'
    if conf_args['channel']['type'] == 'none':
        tb_name += conf_args['channel']['type']
    else:
        tb_name += conf_args['channel']['type'] + \
            '{:0.2f}'.format(conf_args['channel']['chan_param'])
    if conf_args['binarization_off']:
        tb_name += '-bo'
    tb_name +=  "-tx" + str(conf_args['numb_tx_bits'])
    if conf_args['deep_encoding']:
        tb_name+= '-de'
        
    if conf_args['variable_encoding']:
        tb_name += ('-v2' if conf_args['variable_encoding']==2 else '-v1')
        
    tb_name += '-'+conf_args['add_name']
    return tb_name


def parse_args(arg_to_parse = None):
    """ Function parses args passed in command line
    """
    parent_dir, _ = os.path.split(os.getcwd())
    
    parser = argparse.ArgumentParser(description='Joint Source Channel Coding')
    parser.add_argument('--variable_encoding','-v',default=0,type=int,
        help='0 is no variable, 1 is variable with , 2 is exp method')
    parser.add_argument('--task','-t',default='train',choices=['train','test','beam'])
    parser.add_argument('--summarize','-s',action='store_true')
    parser.add_argument('--dataset','-d',default='giga',choices=['wiki','news','euro','beta','giga'])
    parser.add_argument('--channel','-c',default='erasure',choices=['erasure','awgn','bsc','none'])
    parser.add_argument('--chan_param','-cp',default=0.95,type=float,help='Keep rate or sig value of channel')
    parser.add_argument('--numb_epochs','-e',default=10,type=int)
    parser.add_argument('--deep_encoding','-de',action='store_true')
    parser.add_argument('--deep_encoding_params','-dp',nargs='+',type=int,
        default=[1000,800,600], help='dim of additional dense layers after lstm in enc')
    parser.add_argument('--lr','-lr',default=0.001,type=float,help='learning rate')
    parser.add_argument('--lr_dec','-lrd',default=2,type=float,
        help='How much to decrease learning rate if validation acc does not improve')
    parser.add_argument('--help_prob','-hp',nargs=2,default=[8,0.02], type=float,
        help='Start of help prob and rate of decreasing help prob')
    parser.add_argument('--numb_tx_bits','-ntx',default=400,type=int)
    parser.add_argument('--binarization_off','-bo',action='store_true',
        help='Switches off the binarization')
    parser.add_argument('--vocab_size','-vs',default=40000,type=int)
    parser.add_argument('--vocab_out','-vo',default=20000,type=int)
    parser.add_argument('--embedding_size','-es',default=200,type=int)
    parser.add_argument('--enc_hidden_units','-eu',default=256,type=int)
    parser.add_argument('--numb_enc_layers','-nel',default=2,type=int)
    parser.add_argument('--numb_dec_layers','-ndl',default=2,type=int)
    parser.add_argument('--batch_size','-b',default=512,type=int)
    parser.add_argument('--batch_size_test','-bt',default=512,type=int)
    parser.add_argument('--min_len','-mil',default=4,type=int)
    parser.add_argument('--max_len','-mal',default=30,type=int)
    parser.add_argument('--diff','-df',default=4,type=int)
    parser.add_argument('--bits_per_bin','-bb',nargs='+',type=int)
    parser.add_argument('--bits_per_bin_gen','-bg',nargs='+',default=['linear',250],
        help='Generates bits per bin. const linear or sqrt followed by low_lim on bits')
    parser.add_argument('--w2n_path','-wp')
    parser.add_argument('--traindata_path','-trp')
    parser.add_argument('--testdata_path','-tep')
    parser.add_argument('--embed_path','-ep')
    parser.add_argument('--model_save_path','-mp')
    parser.add_argument('--model_save_path_initial','-mpi')
    parser.add_argument('--summ_path','-sp')
    parser.add_argument('--log_path','-lp',help='where the logging is done')
    parser.add_argument('--test_results_path','-terp')
    parser.add_argument('--print_every','-pe',default=500,type=int)
    parser.add_argument('--max_test_counter','-mt',default=int(60000),type=int)
    parser.add_argument('--max_validate_counter','-mv',default=1000,type=int)
    parser.add_argument('--max_batch_in_epoch','-mb',default=int(1e6),type=int)
    parser.add_argument('--summary_every','-sme',default=20,type=int)
    parser.add_argument('--peephole','-p',action='store_false')
    parser.add_argument('--beam_size','-bs',default=10,type=int)
    parser.add_argument('--add_name','-an',default='')
    parser.add_argument('--add_name_results','-anr',default='')
    parser.add_argument('--unk_perc','-up',default=0.2,type=float)
    parser.add_argument('--qcap','-q',default=200,type=int)
    parser.add_argument('--gradient_clip_norm','-gcn',default=5.0,type=float)
    
    if arg_to_parse is None:
        conf_args = vars(parser.parse_args())
    else:
        conf_args = vars(parser.parse_args(arg_to_parse))
    
    if conf_args['deep_encoding'] and conf_args['variable_encoding']==1:
        raise ValueError('deep encoding and variable encoding of type 1 are not compatible')
        
    conf_args['channel'] = {'type':conf_args['channel'],'chan_param':conf_args['chan_param']}
    conf_args['help_prob'] = {'start':conf_args['help_prob'][0],
                              'rate':conf_args['help_prob'][1]}
    if conf_args['variable_encoding'] and (conf_args['bits_per_bin'] is None):
        #Generating the bit allocation per bin
        type_bit_bin = conf_args['bits_per_bin_gen'][0]
        if len(conf_args['bits_per_bin_gen'])>1:
            low_lim = int(conf_args['bits_per_bin_gen'][1])
        else:
            low_lim=None
            conf_args['bits_per_bin_gen'].append(None)
        conf_args['bits_per_bin']= bin_batch_create(conf_args['numb_tx_bits'],
                         dataset=conf_args['dataset'],
                         type_bit_bin = type_bit_bin,low_lim=low_lim)
      
    cds = conf_args['dataset']
    conf_args['w2n_path'] = conf_args['w2n_path'] or \
        os.path.join(parent_dir,'data',cds,'w2n_n2w_{}.pickle'.format(cds))
    conf_args['testdata_path'] = conf_args['testdata_path'] or \
        os.path.join(parent_dir,'data',cds,'{}_test.dat'.format(cds))
    conf_args['traindata_path'] = conf_args['traindata_path'] or \
        os.path.join(parent_dir,'data',cds,'{}_train.dat'.format(cds))
    conf_args['embed_path'] = conf_args['embed_path'] or \
        os.path.join(parent_dir,'data',cds,'{}_embed_{}.pickle'.format(conf_args['embedding_size'],cds))
    fileName = generate_tb_filename(conf_args)
    if conf_args['model_save_path']:
        conf_args['model_save_path'] = conf_args['model_save_path']
    else:
        try:
            os.mkdir(os.path.join(parent_dir,'trained_models',cds,fileName))
        except FileExistsError:
            pass
        conf_args['model_save_path'] = os.path.join(parent_dir,'trained_models',
                 cds,fileName,fileName) 
    conf_args['summ_path'] = conf_args['summ_path'] or \
        os.path.join(parent_dir,'tensorboard',cds,fileName)
    conf_args['log_path'] = conf_args['log_path'] or \
        os.path.join(parent_dir,'tensorboard',cds,fileName+conf_args['task']+'.log')
    conf_args['test_results_path'] = conf_args['test_results_path'] or \
        os.path.join(parent_dir,'test_results',cds,
            fileName+conf_args['task']+conf_args['add_name_results']+'.out')
        
    return conf_args

   
if __name__ == "__main__":
    pass
    w2n_news = Word2Numb('../data/news/w2n_n2w_news.pickle', vocab_size = 40000)
    b_news = BatchGenerator('../data/news/news_train.dat', w2n_news, mode='sent_std', 
                            batch_size = 512, epochs = 1, unk_perc = 0.2, vocab_out = 20000)
#    parent_dir = os.path.split(os.getcwd())[0]
#    folder_type = 'giga'
#    path_w2n_n2w = os.path.join(parent_dir, 'data', folder_type,'w2n_n2w_{}.pickle'.format(folder_type))
#    path_corpus = os.path.join(parent_dir, 'data', folder_type, '{}_train.dat'.format(folder_type))
#    path_word_embed = os.path.join(parent_dir,'data','wikipedia','200_embed_wiki.pickle')
#    w2n,n2w,e=word_dict_embed(100000,dim=200,path_corpus=path_corpus,path_w2n_n2w=path_w2n_n2w,path_word_embed=path_word_embed)

#    w2numb = Word2Numb(path_w2n_n2w)
#    batch_gen = BatchGenerator(path_corpus,w2numb,diff=4,min_len=5, max_len=50,batch_size=512,mode='summ_std')
#    gen = batch_gen.get_next_batch()
    