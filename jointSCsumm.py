# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:35:52 2018

This script contains the code to run the joint source and channel op with 
the summarized data set

@author: Milind, Nariman
"""

import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pickle
import bisect
import itertools
from preprocess_library import BatchGenerator, Word2Numb, parse_args
from functools import partial
import time
from threading import Thread, Event
import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
     
class Config(object):
    """The model configuration
    """

    PAD = 0
    EOS = 1
    SOS = 2
    UNK = 3

    def __init__(self,
                 model_save_path=None,
                 channel={'type':'erasure','chan_param':0.9},
                 numb_epochs=10,
                 lr=0.001,
                 lr_dec = 2,
                 numb_tx_bits=200, # Could be a list for variable length
                 vocab_size=40000, # including special <PAD> <EOS> <SOS> <UNK>
                 vocab_out=20000,
                 embedding_size=200,
                 enc_hidden_units=256,
                 numb_enc_layers=2,
                 numb_dec_layers=2,
                 batch_size=512,
                 batch_size_test = 128,
                 min_len=5,
                 max_len=50,
                 diff = 50,
                 help_prob={'start':5,'rate':0.05},
                 bits_per_bin = [200,250,300,350,400,450,500],
                 variable_encoding = 0,
                 beam_size = 10,
                 deep_encoding = False,
                 deep_encoding_params = [1000,800,600],
                 binarization_off = False,
                 peephole = True,
                 dataset = 'euro',
                 w2n_path="../data/giga/w2n_n2w_giga.pickle",
                 traindata_path="../data/giga_train.dat",
                 testdata_path="../data/giga/giga_test.dat",
                 embed_path="../data/giga/200_embed_giga.pickle",
                 summ_path='../tensorboard/giga/tbsum',
                 test_results_path='../results/giga/result.out',
                 print_every = 1,
                 max_test_counter = int(1e6),
                 max_validate_counter = 10000,
                 max_batch_in_epoch = int(1e9),
                 summary_every = 20,
                 qcap = 200,
                 gradient_clip_norm=5.0,
                 **kwargs):
        """
        Args:
            model_save_path - path where the model is saved
            channel - dict with keys type [erasure,awgn] and chan_param
            bits_per_bin - ensure this is even and of the same size as the number of batches 
        """
        self.epochs = numb_epochs
        self.lr = lr
        self.lr_dec = lr_dec

        self.vocab_size = vocab_size
        self.vocab_out = vocab_out
        self.embedding_size = embedding_size # length of embeddings
        self.enc_hidden_units = enc_hidden_units
        self.numb_enc_layers = numb_enc_layers
        self.numb_dec_layers = numb_dec_layers
        self.dec_hidden_units = enc_hidden_units * 2
        self.binarization_off = binarization_off
        #batch properties
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.min_len = min_len
        self.max_len = max_len
        self.diff = diff
        if not bits_per_bin:
            self.bits_per_bin = [numb_tx_bits for _ in range(min_len,max_len,diff)] 
        else:
            self.bits_per_bin = bits_per_bin
        self.variable_encoding = variable_encoding
        self.deep_encoding = deep_encoding
        self.deep_encoding_params = deep_encoding_params
        self.beam_size = beam_size
        self.peephole = peephole
        self.channel = channel
        self.numb_tx_bits = numb_tx_bits
        self.w2n_path = w2n_path
        self.traindata_path = traindata_path
        self.testdata_path = testdata_path
        self.embed_path = embed_path
        self.model_save_path = model_save_path
        self.summ_path = summ_path
        self.test_results_path = test_results_path
        self.dataset=dataset
        self.help_prob = help_prob
        
        self.queue_limits = list(range(min_len,max_len,diff))
        self.print_every = print_every
        self.max_test_counter = max_test_counter
        self.max_batch_in_epoch = max_batch_in_epoch
        self.summary_every = summary_every
        self.max_validate_counter = max_validate_counter
        
        self.qcap = qcap
        self.gradient_clip_norm = gradient_clip_norm
        self.kwargs=kwargs

class Embedding(object):
    """The word embeddings used in the encoder and decoder
    """
    def __init__(self,config):
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        if config.embed_path == None:
            self.embeddings = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size],
                                      -1.0, 1.0),
                                      dtype=tf.float32,
                                      name='embed')
        else:
            with open(config.embed_path, 'rb') as fop:
                embeddings = pickle.load(fop)
                self.embeddings = tf.Variable(embeddings[:self.vocab_size], 
                                              dtype=tf.float32,
                                              name='embed')
        self.curr_embeds = None

    def get_embeddings(self,inputs):    
        self.curr_embeds = tf.nn.embedding_lookup(self.embeddings, inputs)
        return self.curr_embeds

class VSEncoder(object):
    """A variable encoder that includes binarization, is ready for summarization
    and partial saves
    """
    def __init__(self,
                 encoder_input, 
                 encoder_input_len, 
                 batch_id,
                 binarization_id,
                 embedding,
                 config):
        """ 
        Args:
            encoder_input - the sentences (padded) that are the inputs to the encoder unit
            encoder_input_len - the length of the sentences
            batch_id - For variable length, each batch of a different size is 
             mapped to a sentence of a different length. This gives the id of the batch
            binarization_id - 0-train binarizer, 1-test_binarizer,2-no binarizer 
            embedding - the embedding matrix. 
            config - includes configuration of length of encoding for each batch_id
        """
        self.enc_input = encoder_input
        self.enc_input_len = encoder_input_len
        self.numb_enc_layers = config.numb_enc_layers
        self.enc_hidden_units = config.enc_hidden_units
        self.embedding = embedding
        self.peephole = config.peephole
        self.batch_size = config.batch_size
        self.binarization_id = binarization_id
        self.batch_id = batch_id 
        self.config = config
        
        self.enc_state_c, self.enc_state_h = self.build_enc_network()
        self.enc_output = self.reduce_size_and_binarize() 
        self.trainable_vars = {'all':tf.global_variables('enc'),
                               'lstm':tf.global_variables('enc_bi'),
                               'db':tf.global_variables('enc_scal'),
                               'deep':tf.global_variables('enc_deep')}
        
        logging.info('Built encoder network. Showing enc_state_ and enc_output')
        logging.info(str(self.enc_state_c))
        logging.info(str(self.enc_output))
        
    def build_enc_network(self):
        """Build the LSTM encoder
        """
        embedded = self.embedding.get_embeddings(self.enc_input)

        lstm_fw_cells = [tf.contrib.rnn.LSTMCell(num_units=self.enc_hidden_units,
                                                 use_peepholes=self.peephole,
                                                 initializer=tf.glorot_uniform_initializer())
                         for _ in range(self.numb_enc_layers) ]
        lstm_bw_cells = [tf.contrib.rnn.LSTMCell(num_units=self.enc_hidden_units,
                                                 use_peepholes=self.peephole,
                                                 initializer=tf.glorot_uniform_initializer())
                         for _ in range(self.numb_enc_layers)]

        (_,efw_state, ebw_state) = \
        tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=lstm_fw_cells,
                                                    cells_bw=lstm_bw_cells,
                                                    inputs=embedded,
                                                    dtype=tf.float32,
                                                    sequence_length=self.enc_input_len,
                                                    scope='enc_bi')
        e_state_c,e_state_h = zip(*[(tf.concat([fw.c,bw.c],axis=-1),
                             tf.concat([fw.h,bw.h],axis=-1))
                             for (fw,bw) in zip(efw_state,ebw_state) ])
        # e_state_c is [(fw1.c,bw1c.),(fw2.c,bw2.c)...]
        return (e_state_c, e_state_h)
    
    def training_binarizer(self, input_layer):
        """Binarizer function used at training
        """
        prob = tf.truediv(tf.add(1.0, input_layer), 2.0)
        bernoulli = tf.contrib.distributions.Bernoulli(probs=prob, 
                                                       dtype=tf.float32)
        return 2 * bernoulli.sample() - 1

    def test_binarizer(self, input_layer):
        """Binarizer function used during testing
        """
        ones = tf.ones_like(input_layer,dtype=tf.float32)
        neg_ones = tf.scalar_mul(-1.0, ones)
        return tf.where(tf.less(input_layer,0.0), neg_ones, ones)

    def binarize(self,input_layer):
        """This part of the code binarizes the reduced states. The last line ensure the
        backpropagation gradients pass through the binarizer unchanged
        """
        compare_callable = {tf.equal(self.binarization_id,0):
                                partial(self.training_binarizer, input_layer),
                            tf.equal(self.binarization_id,1):
                                partial(self.test_binarizer, input_layer),
                            tf.equal(self.binarization_id,2):
                                (lambda : input_layer)}
        binarized = tf.case(compare_callable,
                            default=(lambda : input_layer),
                            exclusive=True,
                            name='bin_comp')
        pass_through = tf.identity(input_layer) # this is used for pass through gradient back prop
        return pass_through + tf.stop_gradient(binarized - pass_through )
    
    def scale_down(self,input_layer,output_dim, name='',**kwargs):
        enc_scal_down = tf.layers.Dense(output_dim,
                                        activation=tf.tanh,
                                        name='enc_scal'+name)
        scaled_down_pre = enc_scal_down(input_layer)
        scaled_down_bin = self.binarize(scaled_down_pre)
        return scaled_down_bin    
                          
    def deep_encoding(self,enc_state_concat):
        for de_layers in self.config.deep_encoding_params:
            enc_state = tf.layers.dense(enc_state_concat,de_layers,
                                        activation=tf.nn.relu,
                                        name='enc_deep')
            enc_state_concat = enc_state
        return enc_state
        
    def variable_encoding(self,enc_state):
        """ Performs variable encoding"""
        if self.config.variable_encoding==1: 
            #Separate scaling down matrix for all lengths
            compare_callable = {} #For control flow purposes
            for ind,_ in enumerate(self.config.queue_limits):
                _unp_state_bat = self.scale_down(enc_state,
                                                self.config.bits_per_bin[ind],
                                                name='_v1')
                paddings = tf.constant([[0,0],[0,
                                        int(self.config.bits_per_bin[-1]
                                        -self.config.bits_per_bin[ind])]])

                state_enc_bat = tf.pad(_unp_state_bat,paddings,'CONSTANT')
                compare_callable[tf.equal(self.batch_id,ind)] = lambda : state_enc_bat
                    
            state_reduc = tf.case(compare_callable,
                                       default=partial(tf.zeros,shape=[]),
                                       exclusive=True,
                                       name='_v1')
        
        elif self.config.variable_encoding == 2:
            compare_callable = {}

            _long_state = self.scale_down(enc_state,self.config.bits_per_bin[-1],
                                          name='_v2')
            for ind,_ in enumerate(self.config.queue_limits):
                paddings = tf.constant([[0,0],
                                        [0,int(self.config.bits_per_bin[-1]
                                        -self.config.bits_per_bin[ind])]])
                scaled_down_sel = tf.slice(_long_state,[0,0],
                                           [-1,self.config.bits_per_bin[ind]])
                state_enc_bat = tf.pad(scaled_down_sel,paddings,'CONSTANT')
                compare_callable[tf.equal(self.batch_id,ind)] = lambda : state_enc_bat
                     
            state_reduc = tf.case(compare_callable,
                                       default=partial(tf.zeros,shape=[]),
                                       exclusive=True,
                                       name='enc_state_to_bits_var2')
        return state_reduc
    
    def reduce_size_and_binarize(self):
        """reduces the size of the state according to the
        number of bits and binarizes
        """
        enc_state_concat = tf.concat((self.enc_state_c+self.enc_state_h),axis=1) 
        # bat x hidden_u . 2[fw,bw] . 2 [c,h] . enc_layers        
        # Adding relu layers if needed
        enc_state = enc_state_concat if not self.config.deep_encoding \
                    else self.deep_encoding(enc_state_concat)
        state_reduc = self.scale_down(enc_state,self.config.numb_tx_bits) \
                        if not self.config.variable_encoding else \
                        self.variable_encoding(enc_state) 
        return state_reduc

class Channel(object):
    """The binarization layer of the encoder plus the channel model.
       Currently the channel model is either error free, erasure channel,or is
       the AWGN channel.
    """
    def __init__(self, channel_in, chan_param, config):
        self.channel_in = channel_in
        self.numb_dec_layers = config.numb_dec_layers
        self.config = config
        self.channel = config.channel
        self.chan_param = chan_param
        self.channel_out = self.build_channel()
        logging.info('Built channel')
        logging.info(str(self.channel_out))


    def gaussian_noise_layer(self, input_layer, std, name=None):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0,
                                 stddev=std, dtype=tf.float32, name=name)
        return input_layer + noise

    def test_binarizer(self, input_layer):
        """Binarizer function used during testing
        """
        ones = tf.ones_like(input_layer,dtype=tf.float32)
        neg_ones = tf.scalar_mul(-1.0, ones)
        return tf.where(tf.less(input_layer,0.0), neg_ones, ones)
    
    def build_channel(self):
        """Build the final binarization layer and the channel
        """
        # if no channel, just output the encoder states
        if self.channel['type'] == "none":
            chan_output = self.channel_in
        elif self.channel['type'] == "erasure":
            chan_output = tf.nn.dropout(self.channel_in,
                                         keep_prob=self.chan_param,
                                         name="erasure_chan_dropout_ch")*self.chan_param
                            
        elif self.channel['type'] == "awgn":
            chan_output = self.gaussian_noise_layer(self.channel_in,
                                             std=self.chan_param,
                                             name="awgn_chan_noise")
            
        elif self.channel['type'] == 'bsc':
            chan_output = tf.where(tf.greater(
                        tf.random_uniform(shape=tf.shape(self.channel_in)),
                                              self.chan_param),
                                   self.channel_in,
                                   -self.channel_in)            
        else:
            raise NameError('Channel type is not known.')

        return chan_output

class VSDecoder(object):
    '''This is a simple decoder that does not use attention and uses raw_rnn for decoding.
    During training the the estimated bit is fed-back as the next input. Crucially
    different channel_outputs can result in different length embeddings
    '''
    def __init__(self, 
                 chan_output,
                 dec_targets,
                 dec_lengths,
                 embeddings,
                 batch_id,
                 prob_corr_input,
                 config,
                 beam=False):
        """ Builds the decoder network
        Args:
            chan_output - output of the channel/hidden state to be fed
            dec_targets - the decoded sentence
            dec_lengths - length of the decoded input
            embeddings - embeddings matrix
            batch_id - used for variable encoding
            prob_corr_input - teacher forcing for training. prob p, always feed
                the correct next word from dec_targets. Else feed previously
                decoded word
            beam - boolean to do beam search or not
            config - config file
        """
        self.batch_id = batch_id
        self.dec_lengths = dec_lengths
        self.batch_size = config.batch_size
        self.embeddings = embeddings
        self.peephole = config.peephole
        self.prob_corr_input = prob_corr_input 
        
        self.dec_targets = dec_targets
        self.dec_inputs = tf.pad(dec_targets[:,:-1],
                                 [[0,0],[1,0]],
                                 mode='CONSTANT',
                                 constant_values=config.SOS*tf.ones([],tf.int32))
                                 
        self.numb_dec_layers = config.numb_dec_layers
        self.dec_hidden_units = config.dec_hidden_units
        self.vocab_out = config.vocab_out
        self.config = config
        self.beam = beam
        self.init_state = self.expand_chann_out(chan_output)
        self.build_cells()
        if self.beam:
            self.dec_pred, self.dec_pred_others = self.build_beam_network()
        else:
            self.dec_logits, self.dec_pred = self.build_dec_network()
        self.trainable_vars = {'all': tf.global_variables('dec'),
                               'lstm':tf.global_variables('dec_lstm'),
                               'deep':tf.global_variables('dec_deep'),
                               'db':tf.global_variables('dec_scal')}
        logging.info('Built decoder. Showing init_state, dec_pred')
        logging.info(str(self.init_state))
        logging.info(str(self.dec_pred))
        
        
    def scale_up(self,input_layer,output_dim,name=''):
        """ Takes an encoding number of bits and scales it up to create an init 
        state for the decoder (c,h)
        """
        dec_states=[]
        for ind in ['c','h']:
            dec_scal_up = tf.layers.Dense(output_dim,
                                          activation=tf.nn.relu,
                                          name='dec_scal'+ind+name)
            dec_states.append(dec_scal_up(input_layer))
        return dec_states
                           
    def deep_decoding(self,dec_state_chan):
        for de_layers in self.config.deep_encoding_params[::-1]:
            dec_inp = tf.layers.dense(dec_state_chan,
                                      de_layers,
                                      activation = tf.nn.relu,
                                      name='dec_deep')
            dec_state_chan = dec_inp
        return dec_inp
    
    def variable_decoding(self, channel_decoded):
        if self.config.variable_encoding==1: 
            compare_callable = {}               
            for ind,_ in enumerate(self.config.queue_limits):
                channel_decoded_nz = tf.slice(channel_decoded,
                                              0, self.config.bits_per_bin[ind],
                                              name='dec_slice')
                s_c,s_h = self.scale_up(channel_decoded_nz,
                                        self.dec_hidden_units,
                                        name = '_v1')
                compare_callable[tf.equal(self.batch_id,ind)] = lambda:(s_c,s_h)
                
            def_op = lambda: (tf.zeros([]),tf.zeros([]))
            state_c,state_h = tf.case(compare_callable,
                                      default=def_op,
                                      exclusive=True,
                                      name="dec_v1_case")
        else:
            state_c,state_h = self.scale_up(channel_decoded,
                                            self.dec_hidden_units,
                                            name='_v2')
        return (state_c,state_h)
    
    def expand_chann_out(self, channel_out):
        '''Expand the channel output (first layer of the decoder)
        '''
        # Passing it through relu layers if needed
        channel_decoded = channel_out if not self.config.deep_encoding \
                        else self.deep_decoding(channel_out)
        init_state = []                                
        for i in range(self.numb_dec_layers):
            state_c,state_h = self.scale_up(channel_decoded,self.dec_hidden_units) \
                                if not self.config.variable_encoding else \
                                self.variable_decoding(channel_decoded)
            init_state.append(tf.contrib.rnn.LSTMStateTuple(c=state_c, h=state_h))
        return tuple(init_state)

    def build_cells(self):
        self.out_proj = tf.layers.Dense(self.vocab_out, 
                    kernel_initializer=tf.initializers.random_uniform(-1,1),
                    name='output_proj')
        cells = [tf.contrib.rnn.LSTMCell(num_units=self.dec_hidden_units,
                                         use_peepholes=self.peephole)
                for _ in range(self.numb_dec_layers)]
        self.decLSTM = tf.contrib.rnn.MultiRNNCell(cells)
                    
    def build_dec_network(self):
        training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                            self.embeddings.get_embeddings(self.dec_inputs),
                            self.dec_lengths,
                            self.embeddings.embeddings,
                            sampling_probability=1.0-self.prob_corr_input)
        decoder = tf.contrib.seq2seq.BasicDecoder(self.decLSTM,
                                              helper=training_helper,
                                              initial_state=self.init_state,
                                              output_layer=self.out_proj)
        dec_logits_, _,__ = tf.contrib.seq2seq.dynamic_decode(
                            decoder,
                            maximum_iterations = tf.shape(self.dec_targets)[1],
                            scope = 'dec_lstm')
        dec_logits = dec_logits_.rnn_output
        dec_outputs = tf.cast(tf.argmax(dec_logits, 2), tf.int32)
        return (dec_logits, dec_outputs)
    
    def build_beam_network(self):
        self.beam_size = self.config.beam_size
        bd_initial_state = tf.contrib.seq2seq.tile_batch(
                    self.init_state, self.beam_size)
        bdec = tf.contrib.seq2seq.BeamSearchDecoder(self.decLSTM,
                                            self.embeddings.embeddings,
                                            self.dec_inputs[:,0],
                                            self.config.EOS,
                                            bd_initial_state,
                                            self.beam_size,
                                            output_layer=self.out_proj)
        bdec_preds_,_,_ = tf.contrib.seq2seq.dynamic_decode(bdec,
                            maximum_iterations = tf.shape(self.dec_inputs)[1],
                            scope='dec_lstm')
        bdec_preds = bdec_preds_.predicted_ids
        return (tf.cast(bdec_preds[:,:,0],tf.int32),bdec_preds[:,:,1:])       

class VSSystem(object):
    """This generates an end-to-end model that includes the sentence encoder,
    the channel, and the decoder. It also trains the models. variable length 
    embedding. Also has a queue for rapid processing. Is ready for summarization
    """

    def __init__(self, config, train_data, test_data, word2numb,beam=False):
        self.config = config
        self.training_counter = 1
        self.test_counter = 1
        self.train_data = train_data
        self.test_data = test_data
        self.word2numb = word2numb
        self.beam = beam

        # ==== reset graph ====
        tf.reset_default_graph()
        
        # ==== Queue setup ====
        name_dtype_init_shape =[('binarization_id', tf.int32,tf.ones((),dtype=tf.int32),()),
                        ('enc_inputs',tf.int32,tf.zeros((config.batch_size,1),dtype=tf.int32),(None, None)),
                        ('enc_inputs_len',tf.int32,tf.zeros((config.batch_size),dtype=tf.int32),(None,)),
                        ('dec_targets_len',tf.int32,tf.zeros((config.batch_size),dtype=tf.int32),(None,)),
                        ('dec_targets',tf.int32,tf.zeros((config.batch_size,1),dtype=tf.int32),(None, None)),
                        ('helper_prob',tf.float32,tf.ones(()),()),
                        ('chan_param',tf.float32,tf.ones(()),()),
                        ('lr',tf.float32,tf.ones(())*self.config.lr,()),
                        ('batch_id',tf.int32,tf.zeros((),dtype=tf.int32),())]
        names_q,dtype_q,init_q,shape_q = zip(*name_dtype_init_shape)
        self.queue_vars = dict((name,tf.placeholder_with_default(init,shape))
                                for name,_,init,shape in name_dtype_init_shape)
        self.queue = tf.RandomShuffleQueue(self.config.qcap,2,dtype_q,names=names_q)
        self.enqueue_op = self.queue.enqueue(self.queue_vars)
        self.close_queue = self.queue.close(cancel_pending_enqueues=True) 
        self.dequeue_vars = self.queue.dequeue()     
        self.epochs = 0

        # ==== Aliases for placeholders ====
        self.binarization_id = tf.placeholder_with_default(self.dequeue_vars['binarization_id'],
                                      shape=(), name='binarization_id')                     
        self.enc_inputs = tf.placeholder_with_default(self.dequeue_vars['enc_inputs'],
                                      shape=(config.batch_size, None), name='encoder_inputs')
        self.enc_inputs_len = tf.placeholder_with_default(self.dequeue_vars['enc_inputs_len'],
                                      shape=(None,), name='enc_inputs_len')                   
        self.dec_targets_len = tf.placeholder_with_default(self.dequeue_vars['dec_targets_len'],
                                      shape=(None,), name='dec_targets_len')
        self.dec_targets = tf.placeholder_with_default(self.dequeue_vars['dec_targets'],
                                      shape=(config.batch_size, None), name='dec_targets')
        self.helper_prob = tf.placeholder_with_default(self.dequeue_vars['helper_prob'],
                                      shape=(), name='helper_prob')
        self.chan_param = tf.placeholder_with_default(self.dequeue_vars['chan_param'],
                                      shape=(), name='chan_param')
        self.lr = tf.placeholder_with_default(self.dequeue_vars['lr'],
                                      shape=(), name='lr')
        self.batch_id = tf.placeholder_with_default(self.dequeue_vars['batch_id'],
                                      shape=(), name='batch_id')
        
        # ==== Building neural network graph ====
        self.embeddings = Embedding(self.config)

        self.encoder = VSEncoder(self.enc_inputs, 
                                 self.enc_inputs_len, 
                                 self.batch_id,
                                 self.binarization_id,
                                 self.embeddings, 
                                 self.config)
 
        self.channel = Channel(self.encoder.enc_output,
                               self.chan_param,
                               self.config)
        
        self.decoder = VSDecoder(self.channel.channel_out,
                                 self.dec_targets,
                                 self.dec_targets_len,
                                 self.embeddings,
                                 self.batch_id,
                                 self.helper_prob,
                                 self.config,
                                 self.beam)
        
        # ==== define loss and training op and accuracy ====
        if not self.beam:
            self.loss, self.train_op = self.define_loss()
        self.accuracy = self.define_accuracy()

        # ==== set up saving, tensorboard ====
        self._setup_savers()
        
        logging.info('Set up the system')

    def _setup_savers(self):
        self.saver = {}
        self.saver['all'] = tf.train.Saver(max_to_keep=3)
        self.saver['embed'] = tf.train.Saver(var_list=[self.embeddings.embeddings])
        self.saver['lstm'] = tf.train.Saver(var_list=self.encoder.trainable_vars['lstm']+
                                                     self.decoder.trainable_vars['lstm'])
        self.saver['db'] = tf.train.Saver(var_list=self.encoder.trainable_vars['db']+
                                                   self.decoder.trainable_vars['db'])
        if self.encoder.trainable_vars['deep']:
            self.saver['deep'] = tf.train.Saver(var_list=self.encoder.trainable_vars['deep']+
                                                     self.decoder.trainable_vars['deep'])

    def _setup_tb(self):
        tf.summary.scalar("CrossEntLoss", self.loss)
        tf.summary.scalar('lr',self.lr)
        tf.summary.scalar('help_prob',self.helper_prob)
        tf.summary.histogram('global_norm',self.global_norm)
        tf.summary.histogram("enc_state_c", tf.concat(self.encoder.enc_state_c,axis=-1))
        tf.summary.histogram("enc_state_h", tf.concat(self.encoder.enc_state_h,axis=-1))
        tf.summary.histogram('enc_out',self.encoder.enc_output)
        tf.summary.histogram('channel_out',self.channel.channel_out)
        tf.summary.histogram("dec_init", self.decoder.init_state[0][0])
        self.tb_summary = tf.summary.merge_all()
        self.tb_val_summ = tf.summary.scalar("Validation_Accuracy", self.accuracy)
            
    def load_trained_model(self, sess, saved_path,saver_type='all'):
        """
        Loads a trained model from what was saved. Insert the trained model path
        """
        try:
            self.saver[saver_type].restore(sess,saved_path)
            logging.info('Loaded {} from {}'.format(saver_type,saved_path))
        except:
            logging.info('Error loading {} from {}'.format(saver_type,saved_path))
            
    def save_trained_model(self, sess, saved_path, global_step=None, saver_type='all'):
        self.saver[saver_type].save(sess,
                                    saved_path,
                                    global_step=global_step,
                                    write_meta_graph=False)
        logging.info('Saved {} to {}'.format(saver_type,saved_path))
        
    def define_accuracy(self):
        max_len = tf.shape(self.dec_targets)[1]
        seq_mask = tf.sequence_mask(self.dec_targets_len,max_len,dtype=tf.float32)
        eq_indicator = tf.cast(tf.equal(self.decoder.dec_pred, self.dec_targets), dtype=tf.float32)
        accuracy = tf.reduce_sum(seq_mask*eq_indicator)/tf.reduce_sum(seq_mask)
        return accuracy

    def define_loss(self):
        max_len = tf.shape(self.dec_targets)[1]
        seq_mask = tf.sequence_mask(self.dec_targets_len,max_len,dtype=tf.float32)
        
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(self.dec_targets, depth=self.config.vocab_out, dtype=tf.float32),
            logits=self.decoder.dec_logits,)
        # loss function
        loss = tf.reduce_sum(stepwise_cross_entropy*seq_mask)/tf.reduce_sum(seq_mask)
#        loss = tf.reduce_mean(stepwise_cross_entropy)
        
        # train it
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, self.global_norm = tf.clip_by_global_norm(gradients, 
                                                self.config.gradient_clip_norm)
        train_op = optimizer.apply_gradients(zip(gradients, variables))
#        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        return loss, train_op

    def batch_to_feed(self, inputs, max_seq_len=None):
        """
        Creates the next zero padded batch
        Args:
            inputs: list of sentences (integer lists)
            max_seq_len: integer specifying how large should `max_time` 
                dimension be. If None, maximum sequence length would be used

        Outputs:
            batch_out: config.PAD padded batch
            sequence_lengths: sentence len
        """
        sequence_lengths = [len(seq) for seq in inputs]
        max_seq_len = max_seq_len or max(sequence_lengths)
        padded_sequence = [seq+[self.config.PAD]*(max_seq_len-len(seq)) 
                            for seq in inputs ]
        batch_out = np.array(padded_sequence,dtype=None)
        return batch_out, sequence_lengths
    
    def next_feed(self,
                  batch, 
                  binarization_id=0,
                  help_prob=1.0,
                  lr=None,
                  chan_param=None,
                  in_queue=True):
        """
        Generate the data feed from the batch for the queue
        
        Args:
            batch - list of either (sent,sent) or (sentence,summary)
            binarization_id - 0: training bin, 1: test_bin, 2: no bin
            help_prob - teacher forcing probability
            lr - learning rate
            chan_param - changing channel parameter
            in_queue - whether to place batch in queue (training) or for 
                immediate execution in testing/beam decoding
                
        Returns:
            fd - feeddict either to place in queue or for immediate execution
        """
        lr_ = lr or self.config.lr
        chan_param_ = chan_param or self.config.channel['chan_param']
        try:
            # For (sentence,summ)
            batch1, batch2 = zip(*batch)
        except:
            raise ValueError('batch is not of the right type (input,output)')
            
        enc_inputs_, enc_inputs_len_ = self.batch_to_feed(
            [(sequence) + [self.config.EOS] for sequence in batch1])
        dec_targets_, dec_targets_len_ = self.batch_to_feed(
            [(sequence) + [self.config.EOS] + [self.config.PAD] for sequence in batch2])
        if self.config.variable_encoding:
            batch_id_ =  bisect.bisect(self.config.queue_limits,enc_inputs_len_[0])-1
        else:
            batch_id_ = 0
        fd_pre = {'binarization_id': binarization_id,
                'enc_inputs': enc_inputs_,
                'enc_inputs_len': enc_inputs_len_,
                'dec_targets': dec_targets_,
                'dec_targets_len':dec_targets_len_,
                'helper_prob': help_prob,
                'chan_param': chan_param_,
                'lr': lr_,
                'batch_id': batch_id_}
        if in_queue:
            fd = dict((self.queue_vars[name_v],val_v) for name_v,val_v in fd_pre.items())
        else:
            fd = dict((self.dequeue_vars[name_v],val_v) for name_v,val_v in fd_pre.items())
        return fd
    
    def get_help_prob(self):
        """ based on linear annealing of helping probability. Set in config"""
        if self.epochs<self.config.help_prob['start']:
            return 1.0
        else:
            return max(0,1.0-self.config.help_prob['rate']
                       *(self.epochs-self.config.help_prob['start']))
                       
    def enqueue_func(self,coord,sess,new_epoch):
        """ This is the function run by the thread responsible for filling in the 
        queue. Runs the enqueueing op that fills the queue with a placeholder that
        gets filled. 
        
        Args:
            coord - coordinator that does housekeeping on threads
            sess - tensorflow session
            new_epoch - threading.Event object to flag if new epoch is started
        Returns:
            None
        """    
        binarization_id = 2 if self.config.binarization_off else 0 
        try:
            while self.epochs < self.config.epochs:
                for batch_ in itertools.islice(self.train_data.get_next_batch(),
                                               self.config.max_batch_in_epoch):
                    fd = self.next_feed(batch_,
                                        help_prob=self.get_help_prob(),
                                        lr = self._lr,
                                        binarization_id=binarization_id)
                    sess.run(self.enqueue_op,feed_dict=fd)
                    if coord.should_stop(): break 
                else:
                    self.epochs += 1
                    new_epoch.set()
                    continue
                break
        except Exception as e:
            logging.info('ERROR in feeding queue {}'.format(e))
            sess.run(self.close_queue)
            coord.request_stop()
        
    def train(self, sess, train_op=None, should_load=True):
        """
        This trains the network. it uses a queue mechanism to feed in placement
        """
        params = tf.trainable_variables()
        num_params = sum(
            map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        logging.info('Training, model parameters: {}'.format(num_params))
        
        self.epochs = 0
        self.training_counter = 1
        self.test_counter = 1
        
        if should_load:
            self.load_trained_model(sess,self.config.model_save_path,
                                    saver_type='all')
        
        self._setup_tb()
        try:
            tb_writer = tf.summary.FileWriter(self.config.summ_path,session=sess)
        except: #tensorflow 1.8
            tb_writer = tf.summary.FileWriter(self.config.summ_path,sess.graph)
            
        self._lr = self.config.lr
        _val_acc = 0
        _bat_count = 0
        
        train_op = train_op or self.train_op
        try:
            coord = tf.train.Coordinator()
            new_epoch = Event()
            t = Thread(target=self.enqueue_func,
                       args=(coord,sess,new_epoch),
                       name='enq_thread')
            t.daemon = True #kills thread when program terminates
            t.start()
        
            # =================== Train on Training Data ======================
            tic = time.time()
            while self.epochs<self.config.epochs:
                if coord.should_stop():
                    logging.info('Coord has requested a break in training')
                    break
                self.training_counter += 1
                (_, loss, tb_summ) = sess.run([train_op, self.loss, self.tb_summary])
                
                if self.training_counter% self.config.summary_every == 0:
                    tb_writer.add_summary(tb_summ, self.training_counter)
                
                _bat_count += 1
                if self.training_counter % self.config.print_every == 0:
                    toc = time.time()
                    logging.info("Epoch: {} #bat {} train time {} loss {}".format(
                        self.epochs + 1,_bat_count, toc - tic,loss))
                    tic = time.time()
                    
                # =============== Validate on Test Data =======================
                if new_epoch.is_set():
                    new_epoch.clear()
                    acc = self.validate(sess,tb_writer)
                    if acc < _val_acc:
                        self._lr /= self.config.lr_dec
                    else:
                        _val_acc = acc
                    _bat_count = 0
                    self.save_trained_model(sess,self.config.model_save_path,
                                            global_step=self.epochs)
                    
            
        except Exception as e:
            logging.info('Stopping in train loop as error has been raised: {}'.format(e))
        finally:
            logging.info('Finished training')
            sess.run(self.close_queue)
            coord.request_stop()
            coord.join([t],stop_grace_period_secs=5)
            
        # =========================== Save the Model ==========================================
        self.save_trained_model(sess, self.config.model_save_path)


    def validate(self, sess,tb_writer):
        """ Function runs a validation for purposes of writing accuracy summary
        and changing the learning rate
        """        
        acc_list = []
        tic = time.time()
        binarization_id = 2 if self.config.binarization_off else 1
        for batch_ in itertools.islice(self.test_data.get_next_batch(),
                                       self.config.max_validate_counter):
            fd = self.next_feed(batch_, binarization_id=binarization_id,
                                help_prob=0.0, in_queue=False)
            einput_, dtarget_, predict_, accu_, tb_summ = \
                sess.run([self.enc_inputs,
                          self.dec_targets,
                        self.decoder.dec_pred,
                         self.accuracy, 
                         self.tb_val_summ], fd)
            tb_writer.add_summary(tb_summ, self.test_counter)
            acc_list.append(accu_)
            self.test_counter += 1
        toc = time.time()
        logging.info("-- Validation Time: {} Accuracy: {}".format(toc - tic,
                                             np.average(acc_list)))
        for j,inp,tar,pred in zip(range(10),einput_,dtarget_, predict_):
            tx = " ".join(self.word2numb.convert_n2w(inp))
            ax = ' '.join(self.word2numb.convert_n2w(tar))
            rx = " ".join(self.word2numb.convert_n2w(pred))
            logging.info('sample {}:'.format(j + 1))
            logging.info('TX: {}'.format(tx))
            logging.info('AX: {}'.format(ax))
            logging.info('RX: {}'.format(rx))
        return np.mean(acc_list)

    def test(self,sess,should_restore=True):
        """ Function to do inference
        """
        if should_restore:
            self.load_trained_model(sess, self.config.model_save_path)

#        acc_list = []
        binarization_id = 2 if self.config.binarization_off else 1
        with open(self.config.test_results_path, 'w', newline='') as file:
            for batch in itertools.islice(self.test_data.get_next_batch(),
                                          self.config.max_test_counter):
    
                fd = self.next_feed(batch, binarization_id=binarization_id,
                                    help_prob = 0.0, in_queue=False)
                
#                predict_, accu_ = sess.run([self.decoder.dec_pred, self.accuracy,], fd)
#                acc_list.append(accu_)
                predict_= sess.run(self.decoder.dec_pred, fd)
                
                for i, (inp, pred) in enumerate(zip(batch, predict_)):
                    tx = " ".join(self.word2numb.convert_n2w(inp[0]))
                    ax = ' '.join(self.word2numb.convert_n2w(inp[1]))
                    rx = " ".join(self.word2numb.convert_n2w(pred))

                    file.write('TX: {}\n'.format(tx))
                    file.write('AX: {}\n'.format(ax))
                    file.write('RX: {}\n'.format(rx))
                    

#            file.write("Average Accuracy: {}\n".format(np.average(acc_list)))
        
    def test_sentence(self, sess, list_sentences):
        """ Accepts a single sentence or list of sentences, returns predictions,
        accuracies, encoding """
        if type(list_sentences) not in [list,tuple]:
            list_sentences = list_sentences,
        binarization_id = 2 if self.config.binarization_off else 1
        batch = [[self.word2numb.convert_w2n(sent.split(' '))]*2 
                     for sent in list_sentences]
        fd = self.next_feed(batch,binarization_id=binarization_id,
                            help_prob=0.0, in_queue=False)
        pred_, acc_, enc_ = sess.run([self.decoder.dec_pred, self.accuracy,
                                          self.encoder.enc_output],fd)
        return (pred_, acc_, enc_)
         

if __name__ == '__main__':
    conf_args = parse_args()
    logging.basicConfig(filename=conf_args['log_path'],filemode='w',level=logging.INFO)
    logging.info('Init and Loading Data...')
    config = Config(**conf_args)
    word2numb = Word2Numb(config.w2n_path,vocab_size = config.vocab_size)
    config.vocab_size= word2numb.vocab_size
    config.vocab_out = min(config.vocab_out,config.vocab_size)
    
    # Setting up input pipelines
    batchmode = 'summ_std' if conf_args['summarize'] else 'sent_std'
    train_sentence_gen = BatchGenerator(config.traindata_path,
                                        word2numb,
                                        mode=batchmode,
                                        batch_size=config.batch_size,
                                        min_len=config.min_len,
                                        max_len=config.max_len,
                                        diff=config.diff,
                                        unk_perc = conf_args['unk_perc'])

    test_sentences = BatchGenerator(config.testdata_path,
                                    word2numb,
                                    mode=batchmode,
                                    batch_size=config.batch_size,
                                    min_len=config.min_len,
                                    max_len=config.max_len,
                                    diff=config.diff,
                                    unk_perc = conf_args['unk_perc'])

    # Bulk of the programme
    train_test = conf_args['task'] # One of 'train','test','beam'
    if train_test == 'train':
        with open(conf_args['model_save_path']+'.config','w') as fop:
            fop.write(str(conf_args))
        logging.info('Building Network...')
        sysNN = VSSystem(config, train_sentence_gen, test_sentences, word2numb)
        logging.info('Start training...')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if conf_args['model_save_path_initial']:
                sysNN.load_trained_model(sess, 
                                         conf_args['model_save_path_initial'])
            sysNN.train(sess)
            
    elif train_test == 'test':
        logging.info('Building Network...')
        sysNN = VSSystem(config, train_sentence_gen, test_sentences, word2numb)
        
        logging.info('Start testing...')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sysNN.test(sess)
        logging.info('Finished testing...')
        
    elif train_test == 'beam':
        logging.info('Building beam search network')
        beam_size = conf_args['beam_size']
        beam_sys = VSSystem(config, train_sentence_gen, test_sentences,word2numb,
                            beam=True)
        logging.info('Beginning beam search processing')
        logging.info('Start beam decode...')
        with tf.Session() as sess:
            beam_sys.test(sess)        