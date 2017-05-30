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


class Config(object):
    feature_size = 50
    hidden_size = 100
    batch_size = 32
    n_epochs = 2
    lr = 0.001
    dropout = 0.5
    print_after_batchs = 100
    max_grad_norm = 5.
    clip_gradients = True

    # ===============  Encoder Parameters ==================
    # Size of the biLSTMs at the encoder. ROWS are for
    # fw (row 0) and bw (row1) directions and columns are for each layer
    encoder = 1
    biLSTMsizes = [[200, 200],
                   [200, 200]]

    numb_layers_enc = 2  # number of biLSTM layers at the encoder
    numb_tx_bits = 300  # number of transmission bits

    # ===============  Channel Parameters ==================
    chan_params = {'type': 'erasure', 'keep_prob': 0.99}

    # ===============  Decoder Parameters ==================
    decoder = 2
    rcv_bit_to_num = 800  # the output length of the first dense layer at rcv
    max_time_step = 100  # max numb of words in the sentence
    numb_layers_dec = 2  # number of LSTM layers at the decoder
    LSTMsizes = [200,200]  # The size of each layer of the LSTM decoder
    numb_words = 50004  # the total number of words in the vocabulary. Should be overwritten with embedding size/vocab data

    def __init__(self,
                 embed_path,
                 w2n_path,
                 train_path,
                 val_path,
                 model_path,
                 constant_embeddings=True):
        self.embed_path = embed_path
        self.w2n_path = w2n_path
        self.train_path = train_path
        self.val_path = val_path
        self.model_path = model_path
        self.constant_embeddings = constant_embeddings
        # self.chan_params = {'type': 'erasure', 'prob_erasure': 0.1}


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
        decoder = DecoderMultiLSTM(channel.chan_output,self.sentence_placeholder,self.sentence_len_placeholder,
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
                    logits=self.decoder.dec_output, labels=self.sentence_placeholder))
            tf.summary.scalar("CrossEntLoss", loss)

        return loss

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
    sentences = sentences_inp.copy()
    len_batches = []
    if mode == 1:
        # Shuffle indices
        np.random.shuffle(sentences)
    for ind in range(0, int(len(sentences) / batch_size) * batch_size, batch_size):
        batch = sentences[ind:ind + batch_size]
        sen_len_batch = [len(x) for x in batch]
        max_len_batch = max(sen_len_batch)
        pad_batch = [x + [1] * (max_len_batch - len(x)) for x in batch]
        padded_batches += [(pad_batch, sen_len_batch)]
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
    batches = batches_pads(sentences, 10, 0)
    return batches


if __name__ == '__main__':
    parent_dir, _ = os.path.split(os.getcwd())
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
    batches, batch_lens = batches_pads(train_data, config.batch_size, mode=0)
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