import tensorflow.contrib.keras as tfk
import tensorflow as tf
import numpy as np
from preprocess import UNK_ID, PAD_ID, START_ID, END_ID


class LSTMCellDecoder(tf.contrib.rnn.LSTMCell):
    """ Extends the LSTM Cell with an output layer"""

    def __init__(self, num_units,
                 input_size=None,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 num_proj=None,
                 proj_clip=None,
                 num_unit_shards=None,
                 num_proj_shards=None,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tf.tanh,
                 reuse=None,
                 feature_size=None,
                 embeddings=None):
        """ Main thing is to pass the embeddings and the feature size"""
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        super(LSTMCellDecoder, self).__init__(num_units, input_size, use_peepholes,
                                              cell_clip, initializer or xavier_initializer,
                                              num_proj, proj_clip, num_unit_shards,
                                              num_proj_shards,
                                              forget_bias, state_is_tuple, activation, reuse)
        self.embeddings = embeddings
        self.feature_size = feature_size
        self.Aout = tf.get_variable('lstm_dec_A', shape=[num_units, self.feature_size],
                                    dtype=tf.float32, initializer=xavier_initializer)
        self.bout = tf.get_variable('lstm_dec_b', shape=[self.feature_size], dtype=tf.float32,
                                    initializer=xavier_initializer)

    def __call__(self, inputs,
                 state,
                 scope=None):
        outputs, state = super(LSTMCellDecoder, self).__call__(inputs, state, scope)
        outputs = tf.matmul(outputs, self.Aout) + self.bout
        outputs = tf.matmul(outputs, self.embeddings, transpose_b=True)
        return (outputs, state)


class MultiRNNCellDecoder(tf.contrib.rnn.MultiRNNCell):
    """ Extends the RNN cell to work with output layers"""

    def __init__(self, cells, state_is_tuple=True, feature_size=None, embeddings=None):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        super(MultiRNNCellDecoder, self).__init__(cells, state_is_tuple)
        self.embeddings = embeddings
        self.feature_size = feature_size
        self.Aout = tf.get_variable('multirnn_dec_A', shape=[cells[-1].output_size, self.feature_size],
                                    dtype=tf.float32, initializer=xavier_initializer)
        self.bout = tf.get_variable('multirnn_dec_b', shape=[self.feature_size], dtype=tf.float32,
                                    initializer=xavier_initializer)

    def __call__(self, inputs,
                 state,
                 scope=None):
        outputs, state = super(MultiRNNCellDecoder, self).__call__(inputs, state, scope)
        outputs = tf.matmul(outputs, self.Aout) + self.bout
        outputs = tf.matmul(outputs, self.embeddings, transpose_b=True)

        return (outputs, state)


class Config(object):
    feature_size = 50
    hidden_size = 100
    batch_size = 32
    n_epochs = 20
    lr = 0.001
    dropout = 0.5
    print_after_batchs = 100
    max_grad_norm = 5.
    clip_gradients = True

    # ===============  Batch Reader Params ==================
    batch_reader_param = {'shuffle': True,
                          'bucket': False,
                          'bucket_sizes': [],
                          'extra_out_pad': 5}

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
    LSTMsizes = [200, 200]  # The size of each layer of the LSTM decoder
    numb_words = 50004  # the total number of words in the vocabulary. Should be overwritten with embedding size/vocab data

    def __init__(self,
                 embed_path,
                 w2n_path,
                 train_path,
                 val_path,
                 model_path,
                 training=True,
                 constant_embeddings=True):
        self.embed_path = embed_path
        self.w2n_path = w2n_path
        self.train_path = train_path
        self.val_path = val_path
        self.model_path = model_path
        self.training = training
        self.constant_embeddings = constant_embeddings

        # self.chan_params = {'type': 'erasure', 'prob_erasure': 0.1}


class Encoder(object):
    '''
    This object is an encoder object that generates the encoder neural network.
    Use polymorphism to define a different encoder. This object is used in the
    EndChanDecNN class.
    '''

    def __init__(self, enc_input, input_len, config):
        self.enc_input = enc_input  # encoder input
        self.input_len = input_len  # encoder input length
        # Size of the biLSTMs at the encoder. ROWS are for
        # fw (row 0) and bw (row1) directions and columns are for each layer
        self.biLSTM_sizes = config.biLSTMsizes
        self.batch_size = config.batch_size  # training batch size
        self.numb_layers_enc = config.numb_layers_enc  # number of biLSTM layers at the encoder
        self.numb_tx_bits = config.numb_tx_bits  # number of transmission bits
        self.encoder_output = None
        # self.encoder_output = self.generate_encoder_nn()  # generate encoder neural network

    def generate_encoder_nn(self):
        '''
        This function generates a bidirectional LSTM encoder. The outputs of each layer
        corresponding to the last element of each sequence are concatenated and passed
        through a dense layer with tanh activation to create the bits.
        :return: The output of the encoder
        '''
        varNameScope = "Enc_biLSTM_L1"
        with tf.variable_scope(varNameScope):
            lstm1fw = tf.contrib.rnn.LSTMCell(num_units=self.biLSTM_sizes[0][0], state_is_tuple=True)
            lstm1bw = tf.contrib.rnn.LSTMCell(num_units=self.biLSTM_sizes[1][0], state_is_tuple=True)

            output1, states1 = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm1fw,
                cell_bw=lstm1bw,
                dtype=tf.float32,
                sequence_length=self.input_len,
                inputs=self.enc_input)
            fw1, bw1 = output1
            biLSTMInput = tf.concat([fw1, bw1], axis=-1)
            # print(biLSTMInput)
            lastfw1 = []
            lastbw1 = []
            for i in range(self.batch_size):
                lastfw1.append(tf.slice(fw1, [i, self.input_len[i] - 1, 0], [1, 1, self.biLSTM_sizes[0][0]]))
                lastbw1.append(tf.slice(bw1, [i, self.input_len[i] - 1, 0], [1, 1, self.biLSTM_sizes[1][0]]))

            lastfw1 = tf.reshape(lastfw1, (self.batch_size, self.biLSTM_sizes[0][0]))
            lastbw1 = tf.reshape(lastbw1, (self.batch_size, self.biLSTM_sizes[1][0]))
            biLSTMOutput = tf.concat([lastfw1, lastbw1], axis=-1)
            # print(biLSTMOutput)
        for i in range(self.numb_layers_enc - 1):
            varNameScope = "Enc_biLSTM_L" + str(i + 2)
            with tf.variable_scope(varNameScope):
                lstm_fw = tf.contrib.rnn.LSTMCell(num_units=self.biLSTM_sizes[0][i + 1], state_is_tuple=True)
                lstm_bw = tf.contrib.rnn.LSTMCell(num_units=self.biLSTM_sizes[1][i + 1], state_is_tuple=True)
                output, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw,
                    cell_bw=lstm_bw,
                    dtype=tf.float32,
                    sequence_length=self.input_len,
                    inputs=biLSTMInput)

                fw, bw = output
                biLSTMInput = tf.concat([fw, bw], axis=-1)
                # print(biLSTMInput)
                lastfw = []
                lastbw = []
                for j in range(self.batch_size):
                    lastfw.append(tf.slice(fw1, [j, self.input_len[j] - 1, 0], [1, 1, self.biLSTM_sizes[0][i + 1]]))
                    lastbw.append(tf.slice(bw1, [j, self.input_len[j] - 1, 0], [1, 1, self.biLSTM_sizes[1][i + 1]]))

                lastfw = tf.reshape(lastfw1, (self.batch_size, self.biLSTM_sizes[0][i + 1]))
                lastbw = tf.reshape(lastbw1, (self.batch_size, self.biLSTM_sizes[1][i + 1]))
                biLSTMOutput = tf.concat([biLSTMOutput, lastfw, lastbw], axis=-1)
                # print(biLSTMOutput)

        # self.encoder_output = tfk.layers.Dense(self.numb_tx_bits, activation="tanh")(biLSTMOutput)
        self.encoder_output = tf.layers.dense(inputs=biLSTMOutput, units=self.numb_tx_bits, activation=tf.nn.tanh)

        tf.summary.histogram("EncOutput", self.encoder_output)

        return self.encoder_output


def test_encoder():
    tf.reset_default_graph()

    configs = Config(*([' '] * 5))
    configs.feature_size = 5
    configs.hidden_size = 10
    configs.batch_size = 8
    configs.biLSTMsizes = [[9, 9, 9], [9, 9, 9]]
    configs.numb_layers_enc = 3  # number of biLSTM layers at the encoder
    configs.numb_tx_bits = 30  # number of transmission bits

    len_max_inp = 25
    enc_input = tf.random_normal([configs.batch_size, len_max_inp, configs.feature_size])
    input_len = np.random.randint(int(len_max_inp / 2), len_max_inp, configs.batch_size)
    encoder = EncoderStackedBiLSTM(enc_input, input_len, configs)
    enc_out = encoder.generate_encoder_nn()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(enc_out)
        enc_outr = sess.run(enc_out)

    print(enc_outr)


class EncoderStackedBiLSTM(Encoder):
    def __init__(self, embeddings, enc_input, input_len, config):
        super(EncoderStackedBiLSTM, self).__init__(enc_input, input_len, config)
        self.embeddings = embeddings

    def generate_encoder_nn(self):
        '''
        This function generates a bidirectional LSTM encoder. The outputs of each layer
        corresponding to the last element of each sequence are concatenated and passed
        through a dense layer with tanh activation to create the bits.
        :return: The output of the encoder
        '''
        sentence_embeds = tf.nn.embedding_lookup(self.embeddings, self.enc_input)

        varNameScope = "Enc_biLSTM"
        with tf.variable_scope(varNameScope):
            lstm_fw_cells = [tf.contrib.rnn.LSTMCell(num_units=fw_cell_size, state_is_tuple=True)
                             for fw_cell_size in self.biLSTM_sizes[0]]
            lstm_bw_cells = [tf.contrib.rnn.LSTMCell(num_units=bw_cell_size, state_is_tuple=True)
                             for bw_cell_size in self.biLSTM_sizes[1]]

            biLSTMOutput, biLSTMstateFW, biLSTMstateBW = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=lstm_fw_cells,
                cells_bw=lstm_bw_cells,
                inputs=sentence_embeds,
                dtype=tf.float32,
                sequence_length=self.input_len,
            )

            # extract the last RNN output including c_state and m_state
            final_biLSTMOutput = tf.concat(biLSTMstateFW[-1] + biLSTMstateBW[-1], axis=-1)
            # print(final_biLSTMOutput)

        # Dense layer to numb_tx_bits.
        # self.encoder_output = tfk.layers.Dense(self.numb_tx_bits, activation="tanh")(final_biLSTMOutput)
        self.encoder_output = tf.layers.dense(inputs=final_biLSTMOutput,
                                              units=self.numb_tx_bits,
                                              activation=tf.nn.tanh)

        tf.summary.histogram("EncOutput", self.encoder_output)
        return self.encoder_output


# def extract_axis_1(self, rnn_outputs, data_len):
#        """
#        Get specified elements along the first axis of tensor.
#        :param rnn_outputs: Tensorflow tensor that will be subsetted.
#        :param data_len: The data length for each element in batch (one for each element along axis 0).
#        :return: Subsetted tensor.
#        """
#
#        batch_range = tf.range(tf.shape(rnn_outputs)[0])
#        indices = tf.stack([batch_range, data_len-1], axis=1)
#        res = tf.gather_nd(rnn_outputs, indices)
#
#        return res

class Channel(object):
    '''
    This object creates a channel model. To use a different model use polymorphism
    '''

    def __init__(self, chan_input, config):
        self.chan_params = config.chan_params  # the parameters of the channel
        self.chan_output = self.gen_chan_model(chan_input)

    def gen_chan_model(self, chan_input):
        chanOutput = []
        print(self.chan_params['type'])
        if self.chan_params['type'] == 'erasure':
            print(self.chan_params['keep_prob'])
            chanOutput = tf.nn.dropout(chan_input, keep_prob=self.chan_params['keep_prob'])

        return chanOutput


class Decoder(object):
    '''
    This object creates the decoder NN. To use a different model use polymorphism
    '''

    def __init__(self, dec_input, batch_max_len, config):
        self.rcv_bit_to_num = config.rcv_bit_to_num  # the output length of the first dense layer at rcv
        self.max_time_step = config.max_time_step  # max numb of words in the sentence
        self.numb_layers_dec = config.numb_layers_dec  # number of LSTM layers at the decoder
        self.LSTM_size = config.LSTMsizes  # The size of the LSTM decoder
        self.numb_words = config.numb_words  # the total number of words in the vocabulary
        self.batch_max_len = batch_max_len  # maximum length of the sentence in the batch
        self.dec_input = dec_input
        self.dec_output = None
        self.feature_size = config.feature_size
        self.training = config.training
        # self.dec_output = self.gen_decoder_nn(dec_input)

    def gen_decoder_nn(self):
        # denseOut = tfk.layers.Dense(self.rcv_bit_to_num,
        #                             activation="tanh",name="Dec_Dense_BitToNum")(self.dec_input)

        denseOut = tf.layers.dense(inputs=self.dec_input,
                                   units=self.rcv_bit_to_num,
                                   activation=tf.nn.tanh)

        LSTMinput = tfk.layers.RepeatVector(self.batch_max_len)(denseOut)
        for i in range(self.numb_layers_dec):
            LSTMoutput = tfk.layers.LSTM(self.LSTM_size[i], return_sequences=True)(LSTMinput)
            LSTMinput = LSTMoutput

        self.dec_output = tfk.layers.Dense(self.numb_words, activation="softmax", name="Dec_Softmax")(LSTMoutput)
        return self.dec_output


def test_decoder():
    tf.reset_default_graph()
    len_max_inp = 25

    configs = Config(*([' '] * 5))
    configs.feature_size = 5
    configs.hidden_size = 10
    configs.batch_size = 8
    configs.biLSTMsizes = [[9, 9, 9], [9, 9, 9]]
    configs.numb_layers_enc = 3  # number of biLSTM layers at the encoder
    configs.numb_tx_bits = 30  # number of transmission bits
    configs.LSTMsizes = [9, 9]  # The size of each layer of the LSTM decoder
    configs.rcv_bit_to_num = sum(configs.LSTMsizes) * 2  # the output length of the first dense layer at rcv
    configs.max_time_step = len_max_inp  # max numb of words in the sentence
    configs.numb_layers_dec = 2  # number of LSTM layers at the decoder
    configs.numb_words = 54  # the total number of words in the vocabulary. Should be overwritten with embedding size/vocab data

    embeddings = tf.random_normal([configs.numb_words, configs.feature_size])
    enc_input = tf.random_normal([configs.batch_size, len_max_inp, configs.feature_size])
    input_len = np.random.randint(int(len_max_inp / 2), len_max_inp, configs.batch_size)
    encoder = EncoderStackedBiLSTM(enc_input, input_len, configs)
    dec_inp = encoder.generate_encoder_nn()
    decoder = DecoderMultiLSTM(dec_inp, embeddings, enc_input, input_len, len_max_inp, configs)
    dec_out = decoder.gen_decoder_nn()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        dec_outr = sess.run(dec_out)

    print(dec_outr)
    return dec_outr


class DecoderMultiLSTM(Decoder):
    def __init__(self, dec_input, embeddings, corrSentense, sentence_len, batch_max_len, config):
        super(DecoderMultiLSTM, self).__init__(dec_input, batch_max_len, config)
        self.corrctSentense = corrSentense
        self.sentence_len = sentence_len  # 5 extra pads and 1 start and 1 end of sentence
        self.embeddings = embeddings

    def gen_decoder_nn(self):

        sentence_embeds = tf.nn.embedding_lookup(self.embeddings, self.corrctSentense)

        # denseOut = tfk.layers.Dense(self.rcv_bit_to_num,
        #                             activation="tanh", name="Dec_Dense_BitToNum")(self.dec_input)
        denseOut = tf.layers.dense(inputs=self.dec_input,
                                   units=self.rcv_bit_to_num,
                                   activation=tf.nn.tanh)

        if self.numb_layers_dec == 1:
            cell = LSTMCellDecoder(num_units=self.LSTM_size[0], state_is_tuple=True,
                                   feature_size=self.feature_size, embeddings=self.embeddings)
            init_state = tf.contrib.rnn.LSTMStateTuple(*tf.split(denseOut, num_or_size_splits=2, axis=1))
        if self.numb_layers_dec > 1:
            cells = [tf.contrib.rnn.LSTMCell(num_units=self.LSTM_size[i], state_is_tuple=True)
                     for i in range(self.numb_layers_dec)]
            # print(bottom_cells)
            # top_cell = LSTMCellDecoder(num_units = self.LSTM_size[self.numb_layers_dec-1], state_is_tuple = True,
            #                             feature_size = self.feature_size, embedding = self.embeddings)

            cell = MultiRNNCellDecoder(cells, feature_size=self.feature_size, embeddings=self.embeddings)
            init_state = tf.split(denseOut, num_or_size_splits=self.numb_layers_dec, axis=1)
            init_state = tuple(
                [tf.contrib.rnn.LSTMStateTuple(*tf.split(x, num_or_size_splits=2, axis=1)) for x in init_state])

        if self.training:
            sample_prob = tf.constant(0, dtype=tf.float32, name='sample_prob')
        else:
            sample_prob = tf.constant(1, dtype=tf.float32, name='sample_prob')

        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs=sentence_embeds,
                                                                     sequence_length=self.sentence_len,
                                                                     embedding=self.embeddings,
                                                                     sampling_probability=sample_prob,
                                                                     time_major=False,
                                                                     seed=None,
                                                                     scheduling_seed=None,
                                                                     name=None)

        decoder = tf.contrib.seq2seq.BasicDecoder(helper=helper,
                                                  initial_state=init_state,
                                                  cell=cell,
                                                  output_layer=None)

        dyn_dec_output, final_state = tf.contrib.seq2seq.dynamic_decode(decoder)

        self.dec_output = dyn_dec_output.rnn_output
        # decoder_inter_output = tfk.layers.Dense(self.feature_size, name='dec_rnn_to_embed_size')(
        #     dyn_dec_output.rnn_output)
        # shape_decoder_inter_output = tf.shape(decoder_inter_output)
        # self.dec_output = tf.reshape(
        #     tf.matmul(tf.reshape(decoder_inter_output, [-1, self.feature_size])
        #               , self.embeddings, transpose_b=True),
        #     [shape_decoder_inter_output[0], shape_decoder_inter_output[1], -1])
        print('Here:', self.dec_output)
        return self.dec_output


if __name__ == "__main__":
    test_encoder()
    dec_outr = test_decoder()