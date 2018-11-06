# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:38:42 2018

@author: Milind
"""

import preprocess_library as pl
import jointSCsumm as jsc
import tensorflow as tf
import logging
import sys

#script_args = '-vs 20000 -ntx 400 -d news -pe 1 -c none -b 5 -mb 10 -e 1 -mv 3 -q 5 -mt 10'.split()
script_args = "-d giga -s -mal 45 -up 0.1 -b 256 -mpi ../trained_models/giga/giga-none-bo-tx600/giga-none-bo-tx600 -lrd 1.5".split()
conf_args = pl.parse_args(script_args)

logging.basicConfig(filename=conf_args['log_path'],filemode='w',level=logging.INFO)
logging.info('Init and Loading Data...')
config = jsc.Config(**conf_args)
word2numb = pl.Word2Numb(config.w2n_path,vocab_size = config.vocab_size)
config.vocab_size= word2numb.vocab_size
config.vocab_out = min(config.vocab_out,config.vocab_size)

# Setting up input pipelines
batchmode = 'summ_std' if conf_args['summarize'] else 'sent_std'
train_sentence_gen = pl.BatchGenerator(config.traindata_path,
                                    word2numb,
                                    mode=batchmode,
                                    batch_size=config.batch_size,
                                    min_len=config.min_len,
                                    max_len=config.max_len,
                                    diff=config.diff,
                                    unk_perc = conf_args['unk_perc'])

test_sentences = pl.BatchGenerator(config.testdata_path,
                                word2numb,
                                mode=batchmode,
                                batch_size=config.batch_size,
                                min_len=config.min_len,
                                max_len=config.max_len,
                                diff=config.diff,
                                unk_perc = conf_args['unk_perc'])

# Bulk of the programme

with open(conf_args['model_save_path']+'.config','w') as fop:
    fop.write(str(conf_args))
logging.info('Building Network...')
sysNN = jsc.VSSystem(config, train_sentence_gen, test_sentences, word2numb)
logging.info('Start training...')

train_op_db = tf.train.AdamOptimizer(learning_rate=sysNN.lr).minimize(
                                sysNN.loss, 
                                var_list = sysNN.encoder.trainable_vars['db']
                                + sysNN.decoder.trainable_vars['db'])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sysNN.load_trained_model(sess, conf_args['model_save_path_initial'], 
                             'lstm')
    sysNN.load_trained_model(sess, conf_args['model_save_path_initial'], 
                             'embed')
    sysNN.config.epochs = 3
    sysNN.train(sess, train_op=train_op_db, should_load=False)
    
sysNN = jsc.VSSystem(config, train_sentence_gen, test_sentences, word2numb)   
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sysNN.config.epochs = 10
    sysNN.train(sess)

