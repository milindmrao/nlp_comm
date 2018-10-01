# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:04:59 2018

This is a script to take in raw files from the Gigaword dataset and then 
extract the sentence (paragraph 1) and the headline (summary)

@author: Milind
"""

import os
import re
from tqdm import tqdm
#import itertools
import nltk
#import pickle
import random

class process_giga(object):
    def __init__(self):
        # Initializing the filter
        self._common_words = set(['the',',','.','to','a','of','and','in',"'",'-',
                              's','"','that','for','on','is','it','was','with',
                              'he','as','at','i','by','be','from','have',
                              'his','has','but','are',':','an','they','this',
                              ',"','not','we','who','had','their','you','been',
                              '-','``','#'])
        self._tknzr = nltk.tokenize.WordPunctTokenizer()
        self._re_subn = re.compile(r'[\d]+[\d\.,]*[th]*')
        self._re_subby = re.compile(r' by( [a-z\-\.\']+){2,4}$')
        self._re_subpar = re.compile(r'^(\(.*?\)\s*)+ |\s*\(.*?\)\s*$')
        
    def process_sent_summ(self, sent, summ):
        """ Processes the sentence and summary fixing common noise"""
        # convert to lower case, remove numbers
        sent = self._re_subn.subn(r'#', sent.lower())[0]
        summ = self._re_subn.subn(r'#',summ.lower())[0]
        # Remove byline from the summary
        summ = self._re_subby.sub(r'',summ)
        # Remove beginning parenthesis terms
        summ = self._re_subpar.subn(r'',summ)[0]
        return sent, summ
    
    def filter_sent_summ(self, sent, summ):
        """ Rejects pairs if they are too long or short or have less overlap"""
        len_sent_words = len(sent.split(' '))
        summ_words = self._tknzr.tokenize(summ)
        # Doing checks based on length
        if len_sent_words<=5 or len_sent_words>50 or len(summ.split(' '))<3:
            return False
        # Checking overlap. Rejecting if half the non-stop words in summary not in line
        summ_nonstop = [x for x in summ_words if x not in self._common_words]
        summ_nonstop_sent = [x for x in summ_nonstop if x in sent]
        if len(summ_nonstop_sent)/len(summ_nonstop)<0.5:
            return False
        return True
                                
    def extract_from_gig_file(self, input_file):
        """ This function takes in a filepath and then extracts all the first 
        paragraphs and their summaries and outputs it in the output file"""
        
        re_doc = re.compile(r'<DOC id="[^"]*" type="story"')
        re_enddoc = re.compile(r'</DOC>')
        re_extr = re.compile(r'<HEADLINE>\s+(.*?)\s+</HEADLINE>.*?<P>\s+(.*?)\s+</P>')
        
        with open(input_file, 'r') as ifop:
            for line in ifop:
                if not re_doc.search(line): #This is not a news story
                    continue 
                
                text_article = []
                while(not re_enddoc.search(line) and ifop):
                    line = ifop.readline()[:-1] #Not reading the new line
                    text_article += [line] #Gathering the entire article in one string
                    
                text_article = ' '.join(text_article[:-1])
                summ_sent_extr = re_extr.search(text_article) 
                
                if summ_sent_extr:
                    summ, sent = summ_sent_extr.groups()
                    sent, summ = self.process_sent_summ(sent, summ)
                    if self.filter_sent_summ(sent,summ):
                        yield (sent, summ)
                    
    def extract_from_folders(self, list_inp_folders, path_train, path_test, perc_test=0.2 ):
        """ Takes in a list of folders. Each folder has files to be read. 
        Randomizes files. Reads in a file and randomly writes to train or test"""
        list_files = []
        for inp_folder in list_inp_folders:
            list_files += [os.path.join(inp_folder,x) for x in os.listdir(inp_folder)]
        random.shuffle(list_files)
        
        ftrain = open(path_train,'w')
        ftest = open(path_test,'w')
        for file in tqdm(list_files):
            for sent_summ in self.extract_from_gig_file(file):
                if random.uniform(0,1)<perc_test:
                    ftest.write('{}\n{}\n'.format(*sent_summ))
                else:
                    ftrain.write('{}\n{}\n'.format(*sent_summ))
        ftrain.close()
        ftest.close()
        

#input_folder = ['E:\\nlp_comm\\code\\data\\giga\\giga_test\\']
input_folder = ['/home/milind/data/gigawords/disc1/gigaword_eng_5_d1/data/afp_eng',
                '/home/milind/data/gigawords/disc1/gigaword_eng_5_d1/data/apw_eng']
pg = process_giga()
path_train = '../data/giga/giga_train.dat'
path_test = '../data/giga/giga_test.dat'
pg.extract_from_folders(input_folder,path_train, path_test)

#path_w2n_n2w = 'E:\\nlp_comm\\code\\data\\news\\w2n_n2w_news.pickle'
#with open(path_w2n_n2w,'rb') as pop:
#    w2n, n2w = pickle.load(pop)

                