import tensorflow.contrib.keras as tfk
import tensorflow as tf


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
        #self.encoder_output = self.generate_encoder_nn()  # generate encoder neural network


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


        self.encoder_output = tfk.layers.Dense(self.numb_tx_bits, activation="tanh")(biLSTMOutput)

        tf.summary.histogram("EncOutput", self.encoder_output)

        return self.encoder_output



class EncoderStackedBiLSTM (Encoder):
    def __init__(self, enc_input, input_len, config):
        super(EncoderStackedBiLSTM, self).__init__(enc_input, input_len, config)

    def generate_encoder_nn(self):
        '''
        This function generates a bidirectional LSTM encoder. The outputs of each layer
        corresponding to the last element of each sequence are concatenated and passed
        through a dense layer with tanh activation to create the bits.
        :return: The output of the encoder
        '''
        varNameScope = "Enc_biLSTM"
        with tf.variable_scope(varNameScope):
            lstm_fw_cells = [tf.contrib.rnn.LSTMCell(num_units=fw_cell_size, state_is_tuple=False)
                             for fw_cell_size in self.biLSTM_sizes[0]]
            lstm_bw_cells = [tf.contrib.rnn.LSTMCell(num_units=bw_cell_size, state_is_tuple=False)
                             for bw_cell_size in self.biLSTM_sizes[1]]

            biLSTMOutput, biLSTMstateFW, biLSTMstateBW = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                    cells_fw=lstm_fw_cells,
                                                    cells_bw=lstm_bw_cells,
                                                    inputs=self.enc_input,
                                                    dtype=tf.float32,
                                                    sequence_length=self.input_len,
                                                    )


            print(biLSTMstateFW)
            # extract the last RNN output
            #final_biLSTMOutput = self.extract_axis_1(biLSTMOutput,self.input_len)
            final_biLSTMOutput = tf.concat([biLSTMstateFW[self.numb_layers_enc-1],
                                            biLSTMstateBW[self.numb_layers_enc-1]], axis=-1)
            print(final_biLSTMOutput)


        self.encoder_output = tfk.layers.Dense(self.numb_tx_bits, activation="tanh")(final_biLSTMOutput)

        tf.summary.histogram("EncOutput", self.encoder_output)

        return self.encoder_output

    def extract_axis_1(self, rnn_outputs, data_len):
        """
        Get specified elements along the first axis of tensor.
        :param rnn_outputs: Tensorflow tensor that will be subsetted.
        :param data_len: The data length for each element in batch (one for each element along axis 0).
        :return: Subsetted tensor.
        """

        batch_range = tf.range(tf.shape(rnn_outputs)[0])
        indices = tf.stack([batch_range, data_len-1], axis=1)
        res = tf.gather_nd(rnn_outputs, indices)

        return res

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
            chanOutput = tf.nn.dropout(chan_input,keep_prob=self.chan_params['keep_prob'])

        return chanOutput


class Decoder(object):
    '''
    This object creates the decoder NN. To use a different model use polymorphism
    '''

    def __init__(self, dec_input, batch_max_len,config):
        self.rcv_bit_to_num = config.rcv_bit_to_num  # the output length of the first dense layer at rcv
        self.max_time_step = config.max_time_step  # max numb of words in the sentence
        self.numb_layers_dec = config.numb_layers_dec  # number of LSTM layers at the decoder
        self.LSTM_size = config.LSTMsizes  # The size of the LSTM decoder
        self.numb_words = config.numb_words  # the total number of words in the vocabulary
        self.batch_max_len = batch_max_len # maximum length of the sentence in the batch
        self.dec_input = dec_input
        self.dec_output = None
        #self.dec_output = self.gen_decoder_nn(dec_input)

    def gen_decoder_nn(self):
        denseOut = tfk.layers.Dense(self.rcv_bit_to_num,
                                    activation="tanh",name="Dec_Dense_BitToNum")(self.dec_input)
        LSTMinput = tfk.layers.RepeatVector(self.batch_max_len)(denseOut)
        for i in range(self.numb_layers_dec):
            LSTMoutput = tfk.layers.LSTM(self.LSTM_size[i], return_sequences=True)(LSTMinput)
            LSTMinput = LSTMoutput

        self.dec_output = tfk.layers.Dense(self.numb_words, activation="softmax", name="Dec_Softmax")(LSTMoutput)
        return self.dec_output

class DecoderMultiLSTM(Decoder):
    def __init__(self, dec_input, corrSentense,sentence_len, batch_max_len, config):
        super(DecoderMultiLSTM, self).__init__(dec_input, batch_max_len,config)
        self.corrctSentense =  corrSentense
        self.sentence_len = sentence_len

    def loop_function(self, prev, _):
        return prev

    def gen_decoder_nn(self):
        denseOut = tfk.layers.Dense(self.rcv_bit_to_num,
                                    activation="tanh", name="Dec_Dense_BitToNum")(self.dec_input)

        print(denseOut)
        cell = tf.contrib.rnn.LSTMCell(num_units=self.LSTM_size[0], state_is_tuple=False)
        if self.numb_layers_dec>1:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(
                                            num_units=cell_size, state_is_tuple=False)
                                            for cell_size in self.LSTM_size])

        helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.corrctSentense,
                                                   sequence_length=self.sentence_len)
        print(helper)
        #split0, split1, split2, split3= tf.split(denseOut, num_or_size_splits=self.numb_layers_dec, axis=1)
        #print('split0: ',split0 )
        init_state = denseOut
        if self.numb_layers_dec > 1:
            init_state = tuple(tf.split(denseOut, num_or_size_splits=self.numb_layers_dec, axis=1))
        print(init_state)

        #tf.layers.dense(inputs=)

        #split0, split1= tf.split(denseOut, num_or_size_splits=2, axis=1)
        #init_state = tuple((split0, split1))

        #print(init_state)
        #print(self.corrctSentense)
        decoder = tf.contrib.seq2seq.BasicDecoder(
                                                helper=helper,
                                                initial_state=init_state,
                                                cell=cell,
                                                output_layer=None
                                                )
        print(decoder)
        decoder_output, final_dec_sate = tf.contrib.seq2seq.dynamic_decode(decoder)
        print(decoder_output)