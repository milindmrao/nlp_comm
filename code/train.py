# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:47:40 2017

@author: Milind

All kinds of useful functions and utilities
"""

from preprocess import *
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import nltk
import pickle
import random
import time
from evaluate import exact_match_score, f1_score

class Config(object):
    feature_size = 50
    hidden_size = 100
    batch_size = 32
    n_epochs = 10
    lr = 0.001
    dropout = 0.5
    print_after_batchs = 50
    model_output = 'model.weights'
    max_grad_norm = 5.
    clip_gradients = True
    encoder = 1
    decoder = 2
    def __init__(self,
                 embed_path,
                 train_path,
                 val_path,
                 constant_embeddings=True):
        self.embed_path = embed_path
        self.train_path = train_path
        self.val_path = val_path
        self.constant_embeddings = constant_embeddings


    
class TransmitModel(object):
    def __init__(self,config):
        self.encoder = config.encoder(config)
        self.decoder = config.decoder(config)
        self.config = config
         
        # ==== set up placeholder tokens ========
        self.sentence_placeholder = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name='sentence')
        self.sentence_len_placeholder = tf.placeholder(
            shape=[None], dtype=tf.int32, name='sentence_len')
        self.dropout_placeholder = tf.placeholder(
            shape=[], dtype=tf.float32, name='dropout')

        # ==== assemble pieces ====
        with tf.variable_scope(
                "TM", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_optimizer()

        # ==== set up training/updating procedure ====
        self.saver = tf.train.Saver()

    def setup_system(self):
        """
        This puts the encoder and decoder together and also simulates the dropout
        :return:
        """
        encoder_outputs = self.encoder.encode(self.sentence_embeds, self.sentence_len_placeholder)
        decoder_inputs = tf.nn.dropout(encoder_outputs,self.dropout_placeholder)
        self.sentence_preds = self.decoder.decode(
            decoder_inputs, self.sentence_len_placeholder)

    def setup_optimizer(self):
        # self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(
        #     self.loss)
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        grads, varis = zip(*optimizer.compute_gradients(self.loss))
        grads, varis = list(grads), list(varis)
        if self.config.clip_gradients:
            grads, self.grad_norm = tf.clip_by_global_norm(
                grads, self.config.max_grad_norm)
        else:
            self.grad_norm = tf.global_norm(grads)
        self.train_op = optimizer.apply_gradients(zip(grads, varis))

    def setup_loss(self):
        """
        Loss function. 
        """
        self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        self.sentence_preds, self.sentence_placeholder))

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with open(self.config.embed_path,'rb') as fop:
            embeddings = pickle.load(fop)
            
        if self.config.constant_embeddings:
            self.embeddings = tf.constant(embeddings, dtype=tf.float32)
        else:
            self.embeddings = tf.Variable(embeddings, dtype=tf.float32)
        self.sentence_embeds = tf.nn.embedding_lookup(self.embeddings,
                                               self.sentence_placeholder)
        
    def _create_feed_dict(self,sentence_batch,sentence_len_batch,dropout=1):
        feed_dict = {}
        feed_dict[self.sentence_placeholder] = sentence_batch
        feed_dict[self.sentence_len_placeholder] = sentence_len_batch
        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def optimize(self, session, input_feed):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        output_feed = [self.train_op, self.loss]
        outputs = session.run(output_feed, feed_dict=input_feed)
        return outputs

    def _create_decode_feed_dict(self, sentence,dropout=1):
        feed_dict = {}
        feed_dict[self.sentence_placeholder] = sentence.reshape(
            (1, len(sentence)))
        feed_dict[self.sentence_len_placeholder] = [len(sentence)]
        feed_dict[self.dropout_placeholder]= dropout

    def decode(self, session, sentence):
        """
        Evaluates a single instance
        """
        input_feed = self._create_decode_feed_dict(sentence,self.config.dropout)
        output_feed = [self.sentence_preds]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def answer_nums(self, session, sentence):
        """
        Transmit one sentence
        """
        sentence_softmax = self.decode(session, sentence)
        sentence_num = np.argmax(sentence_softmax,axis = np.shape(sentence_softmax)[-1])
        return sentence_num
    
    def num2word(sentence_num,vocab):
        return list(map(lambda x:vocab[x],sentence_num))

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Our dataset format: a list of (context, question, begin, end)


        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        if len(dataset) > sample:
            dataset = random.sample(dataset, sample)
        f1, em = 0., 0.
        for sentence in dataset:
            sentence_pred_num = self.answer_nums(session,sentence)
            f1 += f1_score(sentence_pred_num, sentence)
            em += exact_match_score(sentence_pred_num, sentence)
        f1 = f1 * 100 / len(dataset)
        em = em * 100 / len(dataset)
        return f1, em

    def _train_on_batch(self, session, sentence_batch, sentence_len_batch):
        input_feed = self._create_feed_dict(
            sentence_batch, sentence_len_batch, self.config.dropout)
        _, loss = self.optimize(session, input_feed)
        return loss

    def _run_epoch(self, session, train_data):
        for i, batch in enumerate(
                batches_pads(train_data, self.config.batch_size),mode=0):
            loss = self._train_on_batch(session, batch[0], batch[1])
            if i == 0:
                print('initial loss: {:f}'.format(loss))
            if (i + 1) % self.config.print_after_batchs == 0:
                print('training loss after {:d} batches is {:f}'.format(i + 1,
                                                                        loss))
        return loss

    def train(self, session, save_train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(
            map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        print('time taken is : %d'%(toc-tic))
        train_data = 2#qa_util.read_train_val_data(self.config.train_path,'train')
        val_data = 1#qa_util.read_train_val_data(self.config.val_path, 'val')
        for epoch in range(self.config.n_epochs):
            print('Begin epoch {:d}'.format(epoch + 1))
            train_cost = self._run_epoch(session, train_data)
            print('Finished epoch {:d}'.format(epoch+1))
            self.saver.save(session,
                            os.path.join(save_train_dir,
                                         self.config.model_output))
            print("Model saved in file: %s" %
                  os.path.join(save_train_dir, self.config.model_output))
            val_cost = self.validate(session, val_data)
            f1, em = self.evaluate_answer(session, val_data)
            print('training loss: {:f}, validation loss: {:f}'.format(
                train_cost, val_cost))
            print('dev_f1: {:f}, dev_em: {:f}'.format(f1, em))

#    def test(self, session, context, question, begin, end):
#        """
#        in here you should compute a cost for your validation set
#        and tune your hyperparameters according to the validation set performance
#        :return:
#        """
#        input_feed = self._create_feed_dict([context], [len(context)],
#                                            [question], [len(question)],
#                                            [begin], [end])
#        output_feed = self.loss
#        outputs = session.run(output_feed, input_feed)
#        return outputs
#    def validate(self, sess, valid_dataset):
#        """
#        Iterate through the validation dataset and determine what
#        the validation cost is.
#
#        This method calls self.test() which explicitly calculates validation cost.
#
#        How you implement this function is dependent on how you design
#        your data iteration function
#
#        :return:
#        """
#        valid_cost = 0
#
#        for context, question, begin, end in valid_dataset:
#            valid_cost = self.test(sess, context, question, begin, end)
#
#        return valid_cost

def dataset_to_token(file_path,word2num):
    """ Function reads a file with a sentence in each line. It converts this to
    a sequence of token numbers and returns this
    
    Inputs:
        file_path : path of the file to read. 
        word2num : from the vocabulary
        
    Returns:
        List of lists. Padding is not applied here. 
        
    Point to note: <unk> is 0. 
    """
    word_token = nltk.tokenize.WordPunctTokenizer()
    sentences = []
    with open(file_path,'rb') as fop:
        sentence_raw = pickle.load(fop)
    
    for single_sentence in sentence_raw:
        tokens = [word2num.get(str.lower(x),0) for x in word_token.tokenize(single_sentence)]
        if sum([x==0 for x in tokens])/len(tokens)<0.2:
            sentences +=[tokens]
#    sentences +=[[word2num.get(str.lower(x),0) for x in word_token.tokenize(y)] for y in sentence_raw]
    return sentences

def batches_pads(sentences_inp, batch_size, mode=0):
    """ Creates batches. 
    Inputs:
        sentences_inp - list of lists
        mode - 0: do nothing, 1: shuffle entries, 2: group sentences of the same size
    Returns:
        list of ([sentence_batch],[sentence_len_batch])
        
    Note:
        <pad> is 1
    """
    padded_batches = []
    sentences =  sentences_inp.copy()
    len_batches = []
    if mode ==1:
        #Shuffle indices
        np.random.shuffle(sentences)
    for ind in range(0,int(len(sentences)/batch_size)*batch_size,batch_size):
        batch = sentences[ind:ind+batch_size]
        sen_len_batch = [len(x) for x in batch]
        max_len_batch = max(sen_len_batch)
        pad_batch = [x+[1]*(max_len_batch-len(x)) for x in batch]
        padded_batches+=[(pad_batch,sen_len_batch)]
        len_batches +=[max_len_batch]
    return padded_batches, len_batches

def test_input():
    # Reading the vocabulary file
    parent_dir,_ = os.path.split(os.getcwd())
    with open(os.path.join(parent_dir,'data','w2n_n2w.pickle'),'rb') as fop:
        [w2n,n2w] = pickle.load(fop)

    # Loading sentences        
    input_file_path = os.path.join(parent_dir,'data','training_reuters.pickle')
    sentences = dataset_to_token(input_file_path,w2n)
    
    # Returning batches
    batches = batches_pads(sentences,10,0)
    
    
if __name__=="__main__":
#    test_input()
#     Reading the vocabulary file
    parent_dir,_ = os.path.split(os.getcwd())
    with open(os.path.join(parent_dir,'data','w2n_n2w.pickle'),'rb') as fop:
        [w2n,n2w] = pickle.load(fop)

    # Loading sentences        
    input_file_path = os.path.join(parent_dir,'data','training_rest_reuters.pickle')
    sentences = dataset_to_token(input_file_path,w2n)
    
#     Returning batches
    batches,lens = batches_pads(sentences,32,0)