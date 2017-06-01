import numpy as np
import os
import tensorflow.contrib.keras as tfk
import tensorflow as tf
import pickle
import time
import nltk
from nnmodels import EncoderStackedBiLSTM
from nnmodels import Channel
from nnmodels import DecoderMultiLSTM
from nnmodels import Config
from preprocess import UNK_ID,PAD_ID,START_ID,END_ID




class EncChanDecNN(object):
    def __init__(self, config):
        self.config = config

        # ==== load num2word and word2num =======
        with open(self.config.w2n_path, 'rb') as fop:
            [self.w2n, self.n2w] = pickle.load(fop)

        # ==== set up placeholder tokens ========
        # self.test_placeholder = tf.placeholder(
        #     shape=[None, None,self.config.feature_size], dtype=tf.float32, name='test')
        self.sentence_placeholder = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name='sentence')
        self.decoder_input = [tf.placeholder(
            shape=[None,1], dtype=tf.float32, name='decoder_input{0}'.format(i)) for i in range(52)]
        self.sentence_len_placeholder = tf.placeholder(
            shape=[None], dtype=tf.int32, name='sentence_len')
        self.lr = tf.placeholder(
            shape=[], dtype=tf.float32, name='lr')
        self.batch_max_len = tf.placeholder(
            shape=[], dtype=tf.int32, name='batch_max_len')

        # ==== assemble pieces ====
        with tf.variable_scope(
                "TM", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.embeddings, self.sentence_embeds = self.setup_embeddings()
            self.encoder, self.channel, self.decoder = self.setup_system()
            self.loss = self.setup_loss()
            self.grad_norm, self.train_op = self.setup_optimizer()

        # ==== set up training/updating procedure ====
        self.saver = tf.train.Saver()
        self.tb_summary = tf.summary.merge_all()

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with open(self.config.embed_path, 'rb') as fop:
            embeddings = pickle.load(fop)

        if self.config.constant_embeddings:
            emb_out = tf.constant(embeddings, dtype=tf.float32)
        else:
            emb_out = tf.Variable(embeddings, dtype=tf.float32)
        sentence_embeds = tf.nn.embedding_lookup(emb_out,
                                                 self.sentence_placeholder)
        return (emb_out, sentence_embeds)

    def setup_system(self):
        """
        This puts the encoder and decoder together and also simulates the dropout
        :return:
        """

        #encoder = Encoder(self.sentence_embeds, self.sentence_len_placeholder, self.config)
        encoder = EncoderStackedBiLSTM(self.sentence_embeds, self.sentence_len_placeholder, self.config)
        encoder.generate_encoder_nn()
        print(encoder.encoder_output)
        channel = Channel(encoder.encoder_output, self.config)

        print(channel.chan_output)
        decoder = DecoderMultiLSTM(channel.chan_output,self.embeddings,self.sentence_embeds,self.sentence_len_placeholder,
                                   self.batch_max_len, self.config)
        decoder.gen_decoder_nn()
        print(decoder.dec_output)
        return (encoder, channel, decoder)

    def setup_loss(self):
        """
        Loss function.
        """
        with tf.name_scope("CrossEntLoss"):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.decoder.dec_output, labels=self.sentence_placeholder)) #Need to add masks
            tf.summary.scalar("CrossEntLoss", loss)

        return loss

	# def setup_loss(self):
        # """
        # Loss function.
        # """
        # with tf.name_scope("CrossEntLoss"):
            # loss = tf.reduce_mean(tf.nn.weighted_moments(
                # tf.nn.sparse_softmax_cross_entropy_with_logits(
                    # logits=self.decoder.dec_output, labels=self.sentence_placeholder),
                        # -1,tf.sequence_mask(self.sentence_len_placeholder,self.batch_max_len))) #Need to add masks
            # tf.summary.scalar("CrossEntLoss", loss)

        # return loss 
		
    def setup_optimizer(self):
        # self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(
        #     self.loss)
        with tf.name_scope("Optimize"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            grads, varis = zip(*optimizer.compute_gradients(self.loss))
            grads, varis = list(grads), list(varis)
            grad_norm = None
            if self.config.clip_gradients:
                grads, self.grad_norm = tf.clip_by_global_norm(
                    grads, self.config.max_grad_norm)
            else:
                grad_norm = tf.global_norm(grads)
            train_op = optimizer.apply_gradients(zip(grads, varis))

        return (grad_norm, train_op)

    def _create_feed_dict(self, sentence_batch, sentence_len_batch, batch_len, lr):
        feed_dict = {}
        feed_dict[self.sentence_placeholder] = sentence_batch
        feed_dict[self.sentence_len_placeholder] = sentence_len_batch
        feed_dict[self.lr] = lr
        feed_dict[self.batch_max_len] = batch_len
        # feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def optimize(self, session, input_feed):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        output_feed = [self.train_op, self.loss, self.tb_summary]
        outputs = session.run(output_feed, feed_dict=input_feed)
        return outputs

    def _train_on_batch(self, session, sentence_batch, sentence_len_batch, batch_len, lr=0.1):
        input_feed = self._create_feed_dict(
            sentence_batch, sentence_len_batch, batch_len, lr)
        _, loss, tb_summary = self.optimize(session, input_feed)
        return (loss, tb_summary)

    def _run_epoch(self, session, train_data, tb_writer):
        batches, batch_lens = batches_pads(train_data, self.config.batch_size, mode=0)
        tic = time.time()
        for i, batch in enumerate(batches):
            lr = 0.01 / np.sqrt(self.tb_itr_counter)
            loss, tb_summary = self._train_on_batch(session, batch[0], batch[1], batch_lens[i], lr)
            tb_writer.add_summary(tb_summary, self.tb_itr_counter)
            self.tb_itr_counter += 1
            if i == 0:
                toc = time.time()
                print('Firts batch training time is : ', (toc - tic))
                print('initial loss: {:f}'.format(loss))
                tic = time.time()
            if (i + 1) % self.config.print_after_batchs == 0:
                toc = time.time()
                print('For this {:d} batch training time is: {:f}'.format(self.config.print_after_batchs,
                                                                          toc - tic))
                print('training loss after {:d} batches is {:f}'.format(i + 1,
                                                                        loss))
        return loss

    def load_data(self):
        """
        This function loads all the training and validation data from file
        :return: train_data: the data used for training
                 valid_data: the data used for validation
        """
        train_data = dataset_to_token(self.config.train_path, self.w2n)
        val_data = dataset_to_token(self.config.val_path, self.w2n)
        return (train_data, val_data)
		
    def decode(self, session, sentence):
        """
        Evaluates a single sentence - can input text and it outputs text
        Inputs:
            session - tensorflow session
            sentence - sentence as a string. 
        Returns:
            The decoded sentence
        """
        word_token = nltk.tokenize.WordPunctTokenizer()
        sentence_batch = [list(map(lambda x:self.w2n.get(x.lower(),0),word_token.tokenize(sentence)))]*self.config.batch_size
        sentence_len_batch = [len(x) for x in sentence_batch]
        batch_len = max(sentence_len_batch)
        input_feed = self._create_feed_dict(sentence_batch, sentence_len_batch, batch_len, self.config.lr)
        output_feed = [self.decoder.dec_output]
        outputs = session.run(output_feed, input_feed)
        outputs_nums = np.argmax(outputs[0][0],axis = -1)
        outputs_words = list(map(lambda x: self.n2w[x],outputs_nums))
        outputs_sentence = ' '.join(outputs_words)
        return outputs_sentence
    
    def load_trained_model(self,sess,trained_model_path):
        """
        Loads a trained model from what was saved. Insert the trained model path
        Inputs:
            trained_model_path (str-path) - Path for storing the training model
        Returns:
            Nothing. Prints if model parameters are loaded or not. 
        """
        trained_model_folder = os.path.split(trained_model_path)[0]
        ckpt = tf.train.get_checkpoint_state(trained_model_folder)
        v2_path = os.path.join(trained_model_folder,os.path.split(ckpt.model_checkpoint_path)[1]+".index")
        norm_ckpt_path = os.path.join(trained_model_folder,os.path.split(ckpt.model_checkpoint_path)[1])
        if ckpt and (tf.gfile.Exists(norm_ckpt_path) or
                     tf.gfile.Exists(v2_path)):
            print("Reading model parameters from %s" %norm_ckpt_path)
            comNN.saver.restore(sess, norm_ckpt_path)
        else:
            print('Error reading weights')

    def train(self, session, tb_writer):
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
        :param tb_writer: the writer for tensorboard
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
        print('Total model parameters: ', num_params)
        print('Loading training and validation data...')
        train_data, val_data = self.load_data()
        print('Finished loading!')
        toc = time.time()
        print('time taken is : %d' % (toc - tic))
        self.tb_itr_counter = 1
        for epoch in range(self.config.n_epochs):
            print('Begin epoch {:d}'.format(epoch + 1))
            train_cost = self._run_epoch(session, train_data, tb_writer)
            print('Finished epoch {:d}'.format(epoch + 1))
            print('Epoch loss: ', train_cost)
            self.saver.save(session, self.config.model_path)
            print("Model saved in file: %s" % self.config.model_path)
            # val_cost = self.validate(session, val_data)
            # f1, em = self.evaluate_answer(session, val_data)
            # print('training loss: {:f}, validation loss: {:f}'.format(
            #    train_cost, val_cost))
            # print('dev_f1: {:f}, dev_em: {:f}'.format(f1, em))


# network = EncChanDecNN(emb_vec_len,enc_params,chan_params,dec_params)

def dataset_to_token(file_path, word2num):
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
    with open(file_path, 'rb') as fop:
        sentence_raw = pickle.load(fop)

    for single_sentence in sentence_raw:
        tokens = [word2num.get(str.lower(x), 0) for x in word_token.tokenize(single_sentence)]
        if sum([x == 0 for x in tokens]) / len(tokens) < 0.2:
            sentences += [tokens]
            #    sentences +=[[word2num.get(str.lower(x),0) for x in word_token.tokenize(y)] for y in sentence_raw]
    return sentences


def batches_pads(sentences_inp, batch_size=32, params={}):
    """ Creates batches for input and output. For the input, it's simply word_tokens followed by padding. 
    For the output, it's <start> word_tokens <end> padding. The longest sentence is also padded 
    Inputs:
        sentences_inp - list of lists
        batch_size - size of each batch
        params- Additional parameters
            'shuffle' (bool): Shuffle the input or not. Default False
            'bucket' (bool): Group by same bucket size. Default False
            'bucket_sizes' (list-of-increasing-int): Bucket boundaries. Default 4:5:40.
            'extra_out_pad' (int): Extra padding for the output. Default 5. 
            
        mode - 0: do nothing, 1: shuffle entries, 2: group sentences of the same size
    Returns:
        list of [(sentence_batch_inp,sentence_batch_out,sentence_len_batch_inp,sentence_len_batch_out)],len_batches

    Note:
        <pad> is PAD_ID
    """
    padded_batches = []
    sentences = sentences_inp.copy()
    len_batches = []
    extra_out_pad = params.get('extra_out_pad',5)
    if params.get('shuffle',False):
        # Shuffle indices so that each batch is different
        np.random.shuffle(sentences)
    if params.get('bucket',False):
        # Tensorflow's method - very experimental
        bucket_sizes = params.get('bucket_sizes',list(range(4,40,5)))
        len_batches,outputs = tf.contrib.training.bucket_by_sequence_length(
                len(sentences_inp),sentences_inp,batch_size,bucket_sizes,dynamic_pad=True)
        padded_batches = [(x,[]) for x in outputs]
    else:
        # Homebrewed method without bucketing
        for ind in range(0, int(len(sentences) / batch_size) * batch_size, batch_size):
            batch = sentences[ind:ind + batch_size]
            sen_len_batch_inp = [len(x) for x in batch]
            max_len_batch = max(sen_len_batch_inp)
            pad_batch_inp = [x + [PAD_ID] * (max_len_batch - len(x)) for x in batch]
            pad_batch_out = [[START_ID]+x+[END_ID]+[PAD_ID]*(max_len_batch - len(x)+extra_out_pad) for x in batch]
            sen_len_batch_out = [x+2 for x in sen_len_batch_inp]
            padded_batches += [(pad_batch_inp, sen_len_batch_inp,pad_batch_out,sen_len_batch_out)]
            len_batches += [max_len_batch]
    return padded_batches, len_batches


def test_input():
    # Reading the vocabulary file
    parent_dir, _ = os.path.split(os.getcwd())
    emb_path = os.path.join(parent_dir, 'data', 'w2n_n2w.pickle')
    with open(os.path.join(parent_dir, 'data', 'w2n_n2w.pickle'), 'rb') as fop:
        [w2n, n2w] = pickle.load(fop)

    # Loading sentences
    input_file_path = os.path.join(parent_dir, 'data', 'training_reuters.pickle')
    sentences = dataset_to_token(input_file_path, w2n)

    # Returning batches
    batches,lens = batches_pads(sentences, 10)
    return (batches,lens)

def setup_loss(logits,labels,lengths,max_sentence_length):
    """
    Loss function.
    """
    with tf.name_scope("CrossEntLoss"):
        loss = tf.reduce_mean(tf.nn.weighted_moments(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels),
                    -1,tf.sequence_mask(lengths,max_sentence_length))) #Need to add masks
    return loss   

	
def test_setup_loss():
    batch_size = 32
    sentences_length = 50
    vocab_size = 200
    sen_lens = np.random.randint(5,sentences_length,[batch_size])
    labs = [np.random.randint(0,vocab_size,x) for x in sen_lens]
    labels = np.array([np.concatenate([x,np.zeros(sentences_length-len(x),dtype=int)],axis=0) for x in labs])
    labels_diff = np.array([np.concatenate([x,np.ones(sentences_length-len(x),dtype=int)],axis=0) for x in labs])
    logits_random = np.random.randn(*[batch_size,sentences_length,vocab_size])
    I_vocab = np.eye(vocab_size)*100
    logits_exact = np.array(list(map(lambda x: I_vocab[x,:],labels)))
    print(np.shape(logits_random),np.shape(labels))
    tf.reset_default_graph()
    loss_random = setup_loss(logits_random,labels,sen_lens,sentences_length)
    loss_random2 = setup_loss(logits_random,labels_diff,sen_lens,sentences_length)
    loss_exact = setup_loss(logits_exact,labels,sen_lens,sentences_length)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        losses = sess.run([loss_random,loss_random2,loss_exact])
    print(losses)
    return losses

if __name__ == '__main__':
    tf.reset_default_graph()
    parent_dir, _ = os.path.split(os.getcwd())
    parent_dir, _ = os.path.split(parent_dir)
    emb_path = os.path.join(parent_dir, 'data', '50_embed.pickle')
    w2n_path = os.path.join(parent_dir, 'data', 'w2n_n2w.pickle')
    train_data_path = os.path.join(parent_dir, 'data', 'training_rest_reuters.pickle')
    valid_data_path = os.path.join(parent_dir, 'data', 'test_reuters.pickle')
    curr_time = str(time.time())
    trained_model_path = os.path.join(parent_dir, 'trained_models', curr_time + 'model.weights')
    summ_path = os.path.join(parent_dir, 'tensorboard', '2')

    print('Building network...')
    config = Config(emb_path, w2n_path, train_data_path, valid_data_path, trained_model_path)
    comNN = EncChanDecNN(config)
    print('Done!')
    train_data,_ = comNN.load_data()
    #print(train_data)
    batches, batch_lens = batches_pads(train_data, config.batch_size)
    print(train_data[0])
    print(train_data[1])
    print(batches[0])
    print(batches[1])

    # print('Start training...')
    # with tf.Session() as sess:
    #     tfk.backend.set_session(sess)
    #     tfk.backend.set_learning_phase(1)
    #     sess.run(tf.global_variables_initializer())
    #
    #     writer = tf.summary.FileWriter(summ_path)
    #     writer.add_graph(sess.graph)
    #     comNN.train(sess, writer)
    # print('Finished training!')