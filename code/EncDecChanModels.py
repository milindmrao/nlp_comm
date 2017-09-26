import tensorflow as tf
import numpy as np
import pickle
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.seq2seq import ScheduledEmbeddingTrainingHelper
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from functools import partial

class Config(object):
    PAD = 0
    EOS = 1
    SOS = 2
    UNK = 3

    def __init__(self,
                 model_save_path,
                 chan_params,
                 numb_epochs=30,
                 lr=0.001,
                 numb_tx_bits=600,
                 vocab_size=19158, # including special <PAD> <EOS> <SOS> <UNK>
                 embedding_size=200,
                 enc_hidden_units=256,
                 numb_enc_layers=2,
                 numb_dec_layers=2,
                 numb_attn_bits = 0,
                 batch_size=128,
                 length_from=4,
                 length_to=30,
                 bin_len = 4,
                 peephole = True,
                 w2n_path="../../data/w2n_n2w_TopEuro.pickle",
                 traindata_path="../../data/training_euro_wordlist30.pickle",
                 testdata_path="../../data/testing_euro_wordlist30.pickle",
                 embed_path="../../data/200_embed_large_TopEuro.pickle"):

        self.epochs = numb_epochs
        self.lr = lr
        self.model_save_path = model_save_path
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size # length of embeddings
        self.enc_hidden_units = enc_hidden_units
        self.numb_enc_layers = numb_enc_layers
        self.numb_dec_layers = numb_dec_layers
        self.dec_hidden_units = enc_hidden_units * 2
        self.batch_size = batch_size
        self.length_from = length_from
        self.length_to = length_to
        self.bin_len = bin_len
        self.peephole = peephole
        self.chan_params = chan_params
        self.numb_tx_bits = numb_tx_bits
        self.numb_attn_bits = numb_attn_bits
        self.w2n_path = w2n_path
        self.traindata_path = traindata_path
        self.testdata_path = testdata_path
        self.embed_path = embed_path


class MyAttentionWrapper(AttentionWrapper):
    """Wraps another `RNNCell` with attention.
    """

    def __init__(self,
                 cell,
                 attention_mechanism,
                 sigma2 = 0.125,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):
        """Construct the `AttentionWrapper`.
        Args:
          cell: An instance of `RNNCell`.
          attention_mechanism: An instance of `AttentionMechanism`.
          attention_layer_size: Python integer, the depth of the attention (output)
            layer. If None (default), use the context as attention at each time
            step. Otherwise, feed the context and cell output into the attention
            layer to generate attention at each time step.
          alignment_history: Python boolean, whether to store alignment history
            from all time steps in the final output state (currently stored as a
            time major `TensorArray` on which you must call `stack()`).
          cell_input_fn: (optional) A `callable`.  The default is:
            `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
          output_attention: Python bool.  If `True` (default), the output at each
            time step is the attention value.  This is the behavior of Luong-style
            attention mechanisms.  If `False`, the output at each time step is
            the output of `cell`.  This is the beahvior of Bhadanau-style
            attention mechanisms.  In both cases, the `attention` tensor is
            propagated to the next time step via the state and is used there.
            This flag only controls whether the attention mechanism is propagated
            up to the next cell in an RNN stack or to the top RNN output.
          initial_cell_state: The initial state value to use for the cell when
            the user calls `zero_state()`.  Note that if this value is provided
            now, and the user uses a `batch_size` argument of `zero_state` which
            does not match the batch size of `initial_cell_state`, proper
            behavior is not guaranteed.
          name: Name to use when creating ops.
        """
        super(MyAttentionWrapper, self).__init__(cell,
                                                 attention_mechanism,
                                                 attention_layer_size,
                                                 alignment_history,
                                                 cell_input_fn,
                                                 output_attention,
                                                 initial_cell_state,
                                                 name)
        self.sigma2 = sigma2
        self.sigma2times2 = tf.constant(2*sigma2,dtype=dtypes.float32)


    def GaussWeights(self, time, cell_batch_size):
        times = tf.cast(tf.range(start=0,limit=tf.shape(self._attention_mechanism.keys)[1]),dtypes.float32)
        timeflt = tf.cast(time,dtypes.float32)
        expterm = tf.div(tf.pow(tf.subtract(times, timeflt), 2), self.sigma2times2)
        gauss = tf.exp(-expterm)
        gauss = tf.reshape(gauss, [1, -1])
        return tf.tile(gauss, [cell_batch_size, 1])



    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_size` outputs).
        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:
          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `DynamicAttentionWrapperState`
             containing the state calculated at this time step.
        """
        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
                [check_ops.assert_equal(cell_batch_size,
                                        self._attention_mechanism.batch_size,
                                        message=error_message)]):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        # alignments = self._attention_mechanism(
        #    cell_output, previous_alignments=state.alignments)


        # This code is added for additing gaussian weights to alignments based
        # on the current position
        if self.sigma2>0:
            #alignments = self._attention_mechanism(
            #    cell_output, previous_alignments=state.alignments)
            gaussWieghts = self.GaussWeights(state.time, cell_batch_size)
            #alignments = tf.multiply(alignments, gaussWieghts)
            alignments = gaussWieghts
        else:
            alignments = self._attention_mechanism(
                cell_output, previous_alignments=state.alignments)


        # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
        expanded_alignments = array_ops.expand_dims(alignments, 1)
        # Context is the inner product of alignments and values along the
        # memory time dimension.
        # alignments shape is
        #   [batch_size, 1, memory_time]
        # attention_mechanism.values shape is
        #   [batch_size, memory_time, attention_mechanism.num_units]
        # the batched matmul is over memory_time, so the output shape is
        #   [batch_size, 1, attention_mechanism.num_units].
        # we then squeeze out the singleton dim.
        attention_mechanism_values = self._attention_mechanism.values
        context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
        context = array_ops.squeeze(context, [1])

        if self._attention_layer is not None:
            attention = self._attention_layer(
                array_ops.concat([cell_output, context], 1))
        else:
            attention = context

        if self._alignment_history:
            alignment_history = state.alignment_history.write(
                state.time, alignments)
        else:
            alignment_history = ()

        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            alignments=alignments,
            alignment_history=alignment_history)

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state




class MyScheduledEmbeddingTrainingHelper(ScheduledEmbeddingTrainingHelper):
    """A training helper that adds scheduled sampling.
    Returns -1s for sample_ids where no sampling took place; valid sample id
    values elsewhere.
    """

    def __init__(self, inputs, sequence_length, embedding, sampling_probability, PAD_embbed,
                 time_major=False, seed=None, scheduling_seed=None, name=None):
        """Initializer.
        Args:
          inputs: A (structure of) input tensors.
          sequence_length: An int32 vector tensor.
          embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`.
          sampling_probability: A 0D `float32` tensor: the probability of sampling
            categorically from the output ids instead of reading directly from the
            inputs.
          time_major: Python bool.  Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
          seed: The sampling seed.
          scheduling_seed: The schedule decision rule sampling seed.
          name: Name scope for any created operations.
        Raises:
          ValueError: if `sampling_probability` is not a scalar or vector.
        """
        super(MyScheduledEmbeddingTrainingHelper, self).__init__(inputs, sequence_length, embedding,
                                                                 sampling_probability,
                                                                 time_major, seed, scheduling_seed, name)

        self.PAD_embbed = PAD_embbed


    def sample(self, time, outputs, state, name=None):
        with ops.name_scope(name, "MyScheduledEmbeddingTrainingHelperSample", [time, outputs, state]):
            # Return -1s where we did not sample, and sample_ids elsewhere
            select_sample_noise = random_ops.random_uniform(
                [self.batch_size], seed=self._scheduling_seed)
            select_sample = (self._sampling_probability > select_sample_noise)
            #sample_id_sampler = categorical.Categorical(logits=outputs)
            sample_ids = math_ops.cast(math_ops.argmax(outputs, axis=-1), dtypes.int32)
            return array_ops.where(
                select_sample,
                sample_ids,
                array_ops.tile([-1], [self.batch_size]))


    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with ops.name_scope(name, "MyScheduledEmbeddingTrainingHelperSample",
                            [time, outputs, state, sample_ids]):
            (finished, base_next_inputs, state) = (
                super(ScheduledEmbeddingTrainingHelper, self).next_inputs(
                    time=time,
                    outputs=outputs,
                    state=state,
                    sample_ids=sample_ids,
                    name=name))

            def maybe_sample():
                """Perform scheduled sampling."""
                where_sampling = math_ops.cast(
                    array_ops.where(sample_ids > -1), dtypes.int32)
                where_not_sampling = math_ops.cast(
                    array_ops.where(sample_ids <= -1), dtypes.int32)
                where_sampling_flat = array_ops.reshape(where_sampling, [-1])
                where_not_sampling_flat = array_ops.reshape(where_not_sampling, [-1])
                sample_ids_sampling = array_ops.gather(sample_ids, where_sampling_flat)
                inputs_not_sampling = array_ops.gather(
                    base_next_inputs, where_not_sampling_flat)
                # print("this is Important****:", inputs_not_sampling)
                sampled_next_inputs = self._embedding_fn(sample_ids_sampling)
                base_shape = array_ops.shape(base_next_inputs)
                return (array_ops.scatter_nd(indices=where_sampling,
                                             updates=sampled_next_inputs,
                                             shape=base_shape)
                        + array_ops.scatter_nd(indices=where_not_sampling,
                                               updates=inputs_not_sampling,
                                               shape=base_shape))

            all_finished = math_ops.reduce_all(finished)
            next_inputs = control_flow_ops.cond(
                all_finished, lambda: self.PAD_embbed, maybe_sample)

            return (finished, next_inputs, state)


class Embedding(object):
    def __init__(self,config):
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        if config.embed_path == None:
            self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                                            dtype=tf.float32)
        else:
            with open(config.embed_path, 'rb') as fop:
                embeddings = pickle.load(fop)
                self.embeddings = tf.Variable(embeddings, dtype=tf.float32)
        self.curr_embeds = None

    def get_embeddings(self,inputs):    # this thing could get huge in a real world application
        self.curr_embeds = tf.nn.embedding_lookup(self.embeddings, inputs)
        return self.curr_embeds


class SimpleEncoder(object):
    def __init__(self,encoder_input, encoder_input_len, embedding,config):
        self.enc_input = encoder_input
        self.enc_input_len = encoder_input_len
        self.numb_enc_layers = config.numb_enc_layers
        self.enc_hidden_units = config.enc_hidden_units
        self.numb_attn_bits = config.numb_attn_bits
        self.embedding = embedding
        self.peephole = config.peephole
        self.batch_size = config.batch_size
        self.W = tf.Variable(tf.random_uniform([self.enc_hidden_units*2, self.numb_attn_bits], -1, 1), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([self.numb_attn_bits]), dtype=tf.float32)
        self.enc_outputs, self.enc_state_c, self.enc_state_h = self.build_enc_network()



    def build_enc_network(self):

        embedded = self.embedding.get_embeddings(self.enc_input)
        # encoder_cell_fw = tf.contrib.rnn.LSTMCell(self.enc_hidden_units)
        # encoder_cell_bw = tf.contrib.rnn.LSTMCell(self.enc_hidden_units)
        #
        # ((encoder_fw_outputs,
        #   encoder_bw_outputs),
        #  (encoder_fw_final_state,
        #   encoder_bw_final_state)) = (
        #     tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
        #                                     cell_bw=encoder_cell_bw,
        #                                     inputs=embedded,
        #                                     sequence_length=self.enc_input_len,
        #                                     dtype=tf.float32, time_major=True)
        # )
        # Concatenates tensors along one dimension.
        # concat_out = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        lstm_fw_cells = [tf.contrib.rnn.LSTMCell(num_units=self.enc_hidden_units,
                                                 use_peepholes=self.peephole,
                                                 state_is_tuple=True)
                         for _ in range(self.numb_enc_layers) ]
        lstm_bw_cells = [tf.contrib.rnn.LSTMCell(num_units=self.enc_hidden_units,
                                                 use_peepholes=self.peephole,
                                                 state_is_tuple=True)
                         for _ in range(self.numb_enc_layers)]

        (enc_outputs,
         encoder_fw_final_state,
          encoder_bw_final_state) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=lstm_fw_cells,
                                                                                    cells_bw=lstm_bw_cells,
                                                                                    inputs=embedded,
                                                                                    dtype=tf.float32,
                                                                                    sequence_length=self.enc_input_len,
                                                                                   )

        # print(enc_outputs)
        print(encoder_fw_final_state)
        # print(encoder_bw_final_state)
        #attenOut = self.attention((encoder_fw_outputs, encoder_bw_outputs),2,time_major=True)
        #print(attenOut)
        concat_out=enc_outputs
        #print(concat_out)


        if self.numb_attn_bits > 0:
            # Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
            # reduces dimensionality
            batch_size, max_steps, dim = tf.unstack(tf.shape(concat_out))
            # flattened output tensor
            outputs_flat = tf.reshape(concat_out, (-1, dim))
            print(outputs_flat)

            # pass flattened tensor through decoder
            reduc_bit_size = tf.add(tf.matmul(outputs_flat, self.W), self.b)
            print(reduc_bit_size)
            encoder_outputs = tf.reshape(reduc_bit_size, shape=(self.batch_size, max_steps, self.numb_attn_bits))
            print(encoder_outputs)
        else:
            encoder_outputs = None

        fw_c_state = encoder_fw_final_state[0].c
        fw_h_state = encoder_fw_final_state[0].h
        bw_c_state = encoder_bw_final_state[0].c
        bw_h_state = encoder_bw_final_state[0].h
        for i in range(1,self.numb_enc_layers):
            fw_c_state = tf.concat([fw_c_state, encoder_fw_final_state[i].c], 1)
            fw_h_state = tf.concat([fw_h_state, encoder_fw_final_state[i].h], 1)
            bw_c_state = tf.concat([bw_c_state, encoder_bw_final_state[i].c], 1)
            bw_h_state = tf.concat([bw_h_state, encoder_bw_final_state[i].h], 1)

        encoder_final_state_c = tf.concat(
            (fw_c_state, bw_c_state), 1)

        encoder_final_state_h = tf.concat(
            (fw_h_state, bw_h_state), 1)

        print(encoder_final_state_c)
        print(encoder_final_state_h)
        # TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
        # encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
        #     c=encoder_final_state_c,
        #     h=encoder_final_state_h
        # )
        return (encoder_outputs, encoder_final_state_c, encoder_final_state_h)

class SimpleChannel(object):
    def __init__(self, enc_out, enc_state_c, enc_state_h, isTrain, keep_rate, config):
        self.enc_out = enc_out
        self.numb_dec_layers = config.numb_dec_layers
        self.enc_state_c = enc_state_c
        self.enc_state_h = enc_state_h
        self.isTrain = isTrain
        self.numb_attn_bits = config.numb_attn_bits
        self.config = config
        self.chan_params = config.chan_params
        self.keep_rate = keep_rate
        self.state_c_reduc = None
        self.state_h_reduc = None
        self.state_c_bits = None
        self.state_h_bits = None
        self.state_c_out = None
        self.state_h_out = None
        # if self.numb_attn_bits>0:
        #     self.W_attn = tf.Variable(tf.random_uniform([self.numb_attn_bits,self.config.dec_hidden_units], -1, 1),
        #                               dtype=tf.float32)
        #     self.b_attn = tf.Variable(tf.random_uniform([self.config.dec_hidden_units], -1, 1),
        #                               dtype=tf.float32)

        #self.attn_out = self.build_attn_chan()
        self.channel_out = self.build_channel()
        print("chan out", self.channel_out)

    def training_binarizer(self, input):
        prob = tf.truediv(tf.add(1.0, input), 2.0)
        bernoulli = tf.contrib.distributions.Bernoulli(probs=prob, dtype=tf.float32)
        return 2 * bernoulli.sample() - 1

    def test_binarizer(self, input):
        ones = tf.ones_like(input,dtype=tf.float32)
        neg_ones = tf.scalar_mul(-1.0, ones)
        return tf.where(tf.less(input,0.0), neg_ones, ones)

    def binarize(self,input):
        # This part of the code binarizes the resuced states. The last line ensure the
        # backpropagation gradients pass through the binarizer unchanged
        binarized = tf.cond(self.isTrain,
                            partial(self.training_binarizer, input),
                            partial(self.test_binarizer, input))

        pass_through = tf.identity(input) # this is used for pass through gradient back prop
        return pass_through + tf.stop_gradient(binarized - pass_through )


    def build_attn_chan(self):
        if self.numb_attn_bits == 0:
            return None

        print("enc_out",self.enc_out )
        if self.chan_params['type'] == "none":
            attn_out = self.enc_out
        elif self.chan_params['type'] == "erasure":
            binarized_attn = self.binarize(self.enc_out)
            attn_out = tf.nn.dropout(binarized_attn,
                                     keep_prob=self.keep_rate,
                                     name="erasure_chan_dropout_attn")
            #reshaped_attn = tf.reshape(attn_out, shape=[-1, self.numb_attn_bits])
            #attn_out = tf.add(tf.matmul(reshaped_attn,self.W_attn),self.b_attn)
            #attn_out = tf.reshape(attn_out, shape=[-1, self.config.batch_size, self.config.dec_hidden_units])
        else:
            raise NameError('Channel type is not known.')

        return attn_out



    def build_channel(self):
        # if no channel, just output the encoder states
        if self.chan_params['type'] == "none":
            self.state_c_out = self.enc_state_c
            self.state_h_out = self.enc_state_h
            channel_out = tf.contrib.rnn.LSTMStateTuple(c=self.state_c_out, h=self.state_h_out)
        elif self.chan_params['type'] == "erasure":
            self.state_c_reduc = tf.layers.dense(inputs=self.enc_state_c,
                                                 units=self.config.numb_tx_bits / 2,
                                                 activation=tf.nn.tanh,
                                                 name="stateC_to_bits")

            self.state_h_reduc = tf.layers.dense(self.enc_state_h,
                                                 self.config.numb_tx_bits / 2,
                                                 activation=tf.nn.tanh,
                                                 name="stateH_to_bits")

            self.state_c_bits = self.binarize(self.state_c_reduc)

            self.state_h_bits = self.binarize(self.state_h_reduc)

            self.state_c_out = tf.nn.dropout(self.state_c_bits,
                                                keep_prob=self.keep_rate,
                                                name="erasure_chan_dropout_c")

            self.state_h_out = tf.nn.dropout(self.state_h_bits,
                                                keep_prob=self.keep_rate,
                                                name="erasure_chan_dropout_h")
        else:
            raise NameError('Channel type is not known.')

        #print(channel_out)
        #channel_out = tf.contrib.rnn.LSTMStateTuple(c=self.state_c_out, h=self.state_h_out)
        return (self.state_c_out, self.state_h_out)

class SimpleDecoder(object):
    '''
    This is a simple decoder that does not use attention and uses raw_rnn for decoding.
    During training the the estimated bit is fed-back as the next input

    '''
    def __init__(self, enc_inputs, encoder_input_len, chan_output, embeddings,
                 dec_inputs, prob_corr_input, config):
        self.enc_input = enc_inputs
        self.enc_input_len = encoder_input_len
        self.decoder_lengths = self.enc_input_len + 3
        self.batch_size = config.batch_size
        self.peephole = config.peephole
        self.prob_corr_input = prob_corr_input
        self.decoder_input = dec_inputs
        #self.one = tf.constant(1.0, dtype=tf.float32)
        self.numb_dec_layers = config.numb_dec_layers
        self.dec_hidden_units = config.dec_hidden_units
        self.vocab_size = config.vocab_size
        self.init_state = self.expand_chann_out(chan_output)

        self.eos_step_embedded = None
        self.sos_step_embedded = None
        self.pad_step_embedded = None
        self.embeddings = embeddings
        # weights and bias for output projection
        self.W = tf.Variable(tf.random_uniform([self.dec_hidden_units, self.vocab_size], -1, 1), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32)


        self.dec_logits, self.dec_pred = self.build_dec_network

    def expand_chann_out(self, channel_out):
        bits_with_erasure_c = channel_out[0]
        bits_with_erasure_h = channel_out[1]
        init_state = []
        for i in range(self.numb_dec_layers):
            self.state_c_out = tf.layers.dense(bits_with_erasure_c,
                                               self.dec_hidden_units,
                                               activation=None,
                                               name="stateC_out_L{}".format(i))

            self.state_h_out = tf.layers.dense(bits_with_erasure_h,
                                               self.dec_hidden_units,
                                               activation=tf.nn.tanh,
                                               name="stateH_out_L{}".format(i))
            init_state.append(tf.contrib.rnn.LSTMStateTuple(c=self.state_c_out, h=self.state_h_out))
        if self.numb_dec_layers==1:
            return init_state[0]
        else:
            return tuple(init_state)

    @property
    def build_dec_network(self):
        '''
        Build the decoder network
        '''
        cell = tf.contrib.rnn.LSTMCell(num_units=self.dec_hidden_units,
                                       use_peepholes=self.peephole,
                                       state_is_tuple=True)
        if self.numb_dec_layers>1:
            cells = [tf.contrib.rnn.LSTMCell(num_units=self.dec_hidden_units,
                                             use_peepholes=self.peephole,
                                             state_is_tuple=True)
                     for _ in range(self.numb_dec_layers)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)

        batch_size, encoder_max_time = tf.unstack(tf.shape(self.enc_input))

        eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
        sos_time_slice = tf.add(tf.ones([batch_size], dtype=tf.int32, name='SOS'), 1)
        pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

        self.eos_step_embedded = self.embeddings.get_embeddings(eos_time_slice)
        self.sos_step_embedded = self.embeddings.get_embeddings(sos_time_slice)
        self.pad_step_embedded = self.embeddings.get_embeddings(pad_time_slice)

        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(cell, self.loop_fn)

        decoder_outputs = decoder_outputs_ta.stack()
        decoder_outputs = tf.transpose(decoder_outputs, perm=[1, 0, 2])
        print(decoder_outputs)


        # Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
        # reduces dimensionality
        decoder_batch_size, decoder_max_steps, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        # flattened output tensor
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        print(decoder_outputs)

        # pass flattened tensor through decoder
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.W), self.b)
        print(decoder_outputs)
        # prediction vals
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_batch_size, decoder_max_steps, self.vocab_size))
        print(decoder_logits)

        # final prediction
        decoder_prediction = tf.argmax(decoder_logits, 2)
        print(decoder_prediction)

        return (decoder_logits, decoder_prediction)



    # we define and return these values, no operations occur here
    def loop_fn_initial(self):
        '''
        The initial condition used to setup the decoder loop in raw_rnn
        :return:    initial_elements_finished,
                    initial_input, initial_cell_state,
                    initial_cell_output,
                    initial_loop_state
        '''
        initial_elements_finished = (0 >= self.decoder_lengths)  # all False at the initial step
        # end of sentence
        initial_input = self.sos_step_embedded
        #last time step's cell state
        initial_cell_state = self.init_state
        # none
        initial_cell_output = None
        # none
        initial_loop_state = None # we don't need to pass any additional information
        return (initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)


    # attention mechanism --choose which previously generated token to pass as input in the next timestep
    def loop_fn_transition(self,time, previous_output, previous_state, previous_loop_state):
        '''
        The main body loop function used in the raw_rnn decoder
        :param time:
        :param previous_output:
        :param previous_state:
        :param previous_loop_state:
        :return:
        '''
        W = self.W
        b = self.b
        embeddings = self.embeddings
        bernoulli = tf.contrib.distributions.Bernoulli(probs=self.prob_corr_input, dtype=tf.float32)
        choose_correct = bernoulli.sample()
        correct_token = self.decoder_input[:, time-1]
        correct_input = embeddings.get_embeddings(correct_token)

        cell_output = previous_output

        print("cell_out!", cell_output)
        def get_next_input():
            # dot product between previous ouput and weights, then + biases
            output_logits = tf.add(tf.matmul(cell_output, W), b)

            # Returns the index with the largest value across axes of a tensor.
            prediction = tf.argmax(output_logits, axis=1)
            # embed prediction for the next input
            next_input = embeddings.get_embeddings(prediction)
            return tf.cond(tf.equal(choose_correct, 1.0), lambda: correct_input, lambda: next_input)

        elements_finished = (time >= self.decoder_lengths)  # this operation produces boolean tensor of [batch_size]
        # defining if corresponding sequence has ended



        # Computes the "logical and" of elements across dimensions of a tensor.
        finished = tf.reduce_all(elements_finished)  # -> boolean scalar
        # Return either fn1() or fn2() based on the boolean predicate pred.
        input = tf.cond(finished, lambda: self.pad_step_embedded, get_next_input)

        state = previous_state
        output = cell_output
        loop_state = None

        return (elements_finished,
                input,
                state,
                output,
                loop_state)

    def loop_fn(self, time, previous_output, previous_state, previous_loop_state):
        '''
        The complete loop function to be used in the raw_rnn decoder
        :param time:
        :param previous_output:
        :param previous_state:
        :param previous_loop_state:
        :return:
        '''
        if previous_state is None:  # time == 0
            assert previous_output is None and previous_state is None
            return self.loop_fn_initial()
        else:
            return self.loop_fn_transition(time, previous_output, previous_state, previous_loop_state)


class SingleStepDecoder(object):
    def __init__(self, embeddings, config, curr_input, state_PH):
        self.batch_size = config.batch_size
        self.peephole = config.peephole

        self.numb_dec_layers = config.numb_dec_layers
        self.dec_hidden_units = config.dec_hidden_units
        self.vocab_size = config.vocab_size

        self.embeddings = embeddings


        self.curr_input = curr_input
        self.curr_state = self.state_place_holder_to_tuple(state_PH)
        print("LSTM CURR STATE: ----->", self.curr_state)

        # weights and bias for output projection
        self.W = tf.Variable(tf.random_uniform([self.dec_hidden_units, self.vocab_size], -1, 1), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32)

        self.cell = self.build_cell()
        self.topk_ids, self.topk_probs, self.new_states = self.build_dec_network()



    def build_cell(self):
        cell = tf.contrib.rnn.LSTMCell(num_units=self.dec_hidden_units,
                                       use_peepholes=self.peephole,
                                       state_is_tuple=True)
        if self.numb_dec_layers>1:
            cells = [tf.contrib.rnn.LSTMCell(num_units=self.dec_hidden_units,
                                             use_peepholes=self.peephole,
                                             state_is_tuple=True)
                     for _ in range(self.numb_dec_layers)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)
        return cell

    def state_place_holder_to_tuple(self,state_PH):
        #l = tf.unstack(state_PH, axis=0)
        rnn_tuple_state = [tf.nn.rnn_cell.LSTMStateTuple(c=state_PH[idx][0], h=state_PH[idx][1])
             for idx in range(self.numb_dec_layers)]

        if self.numb_dec_layers == 1:
            return rnn_tuple_state[0]
        else:
            return tuple(rnn_tuple_state)


    def build_dec_network(self):
        input_emb = self.embeddings.get_embeddings(self.curr_input)
        print(input_emb)
        #curr_state = self.state_place_holder_to_tuple()
        print(self.curr_state)
        outputs, states = tf.nn.static_rnn(cell=self.cell,
                                           inputs=[input_emb],
                                           initial_state=self.curr_state,
                                           dtype=tf.float32)
        print(outputs)
        print(states)
        decoder_outputs = outputs[0]
        #decoder_outputs = tf.transpose(decoder_outputs, perm=[1, 0, 2])
        print(decoder_outputs)

        decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))

        # pass flattened tensor through decoder
        decoder_logits = tf.add(tf.matmul(decoder_outputs, self.W), self.b)
        print(decoder_logits)

        # final prediction
        topk_log_probs, topk_ids = tf.nn.top_k(tf.log(tf.nn.softmax(decoder_logits)), decoder_batch_size * 2)
        #decoder_prediction = tf.argmax(decoder_logits, 2)

        print(states)
        # fw_c_state = encoder_fw_final_state[0].c
        # fw_h_state = encoder_fw_final_state[0].h
        # bw_c_state = encoder_bw_final_state[0].c
        # bw_h_state = encoder_bw_final_state[0].h
        # for i in range(1,self.numb_dec_layers):
        #     fw_c_state = tf.concat([fw_c_state, encoder_fw_final_state[i].c], 1)
        #     fw_h_state = tf.concat([fw_h_state, encoder_fw_final_state[i].h], 1)
        #     bw_c_state = tf.concat([bw_c_state, encoder_bw_final_state[i].c], 1)
        #     bw_h_state = tf.concat([bw_h_state, encoder_bw_final_state[i].h], 1)
        out_states = [(states[idx].c, states[idx].h)
                               for idx in range(self.numb_dec_layers)]
        # for idx in range
        # if self.numb_dec_layers == 1:
        #     out_state = [[states.c], [states.h]]
        # else:
        #     rnn_tuple_state = [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
        #                        for idx in range(self.numb_dec_layers)]

        print(topk_log_probs)
        print(topk_ids)
        return topk_ids, topk_log_probs, out_states




class Hypothesis(object):
    """Defines a hypothesis during beam search."""

    def __init__(self, tokens, log_prob, state):
        """Hypothesis constructor.
        Args:
          tokens: start tokens for decoding.
          log_prob: log prob of the start tokens, usually 1.
          state: decoder initial states.
        """
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state

    def Extend(self, token, log_prob, new_state):
        """Extend the hypothesis with result from latest step.
        Args:
          token: latest token from decoding.
          log_prob: log prob of the latest decoded tokens.
          new_state: decoder output state. Fed to the decoder for next step.
        Returns:
          New Hypothesis with the results from latest step.
        """
        return Hypothesis(self.tokens + [token], self.log_prob + log_prob,
                          new_state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    def __str__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob,
                                                              self.tokens))


class BeamSearch(object):
    """Beam search."""

    def __init__(self, embeddings,
                 input_place_holder,
                 state_place_holder,
                 chanout_place_holder,
                 word2numb,
                 beam_size, config):
        """Creates BeamSearch object.
        Args:
          model: Seq2SeqAttentionModel.
          beam_size: int.
          start_token: int, id of the token to start decoding with
          end_token: int, id of the token that completes an hypothesis
          max_steps: int, upper limit on the size of the hypothesis
        """
        self.word2numb = word2numb
        self._beam_size = beam_size
        self._start_token = config.SOS
        self._end_token = config.EOS
        self._max_steps = config.length_to+3
        self.numb_dec_layers = config.numb_dec_layers
        self.dec_hidden_units = config.dec_hidden_units
        self.input_PH = input_place_holder
        self.state_PH = state_place_holder
        self.chan_out_PH = chanout_place_holder
        self.init_state = self.expand_chann_out(self.chan_out_PH)
        self.SingleStepDecoder = SingleStepDecoder(embeddings, config, self.input_PH, self.state_PH)

    def expand_chann_out(self, channel_out):
        print("beam chan out", channel_out)
        bits_with_erasure_c = channel_out[0]
        print("beam bit c", bits_with_erasure_c)
        bits_with_erasure_h = channel_out[1]
        print("beam bit h", bits_with_erasure_h)
        init_state = []
        for i in range(self.numb_dec_layers):
            self.state_c_out = tf.layers.dense(bits_with_erasure_c,
                                               self.dec_hidden_units,
                                               activation=None,
                                               name="stateC_out_L{}".format(i))

            self.state_h_out = tf.layers.dense(bits_with_erasure_h,
                                               self.dec_hidden_units,
                                               activation=tf.nn.tanh,
                                               name="stateH_out_L{}".format(i))
            #init_state.append(tf.contrib.rnn.LSTMStateTuple(c=self.state_c_out, h=self.state_h_out))
            init_state.append([self.state_c_out, self.state_h_out])
        # if self.numb_dec_layers == 1:
        #     return init_state[0]
        # else:
        #     return tuple(init_state)
        return init_state



    def BeamSearch(self, sess, chan_output, beam_size = None):
        """Performs beam search for decoding.
        Args:
          sess: tf.Session, session
          channel_out: The output of the channel
        Returns:
          hyps: list of Hypothesis, the best hypotheses found by beam search,
              ordered by score
        """
        if beam_size!= None:
            self._beam_size = beam_size
        init_state = sess.run(self.init_state, feed_dict={self.chan_out_PH:chan_output})
        #print(np.array(init_state).shape)

        # Replicate the initial states K times for the first step.
        hyps = [Hypothesis([self._start_token], 0.0, np.array(init_state))
                ] * self._beam_size
        results = []

        steps = 0
        while steps < self._max_steps and len(results) < self._beam_size:
            #print("STEP -------------------->", steps)
            latest_tokens = [h.latest_token for h in hyps]
            #print(latest_tokens.shape)
            curr_states = [h.state for h in hyps]
            #state_input = sess.run(curr_state)
            curr_states = np.array(curr_states)

            #print(curr_states)
            #print(curr_states.shape)
            #curr_states = curr_states.reshape((self.numb_dec_layers, 2, -1, self.dec_hidden_units))
            if steps==0:
                curr_states = np.swapaxes(curr_states,0,3).squeeze(axis=0)
            else:
                curr_states = np.swapaxes(curr_states, 0, 1)
                curr_states = np.swapaxes(curr_states, 1, 2)

            #print(curr_states.shape)
            #print(curr_states[0][0])

            fd = {self.input_PH: latest_tokens,
                  self.state_PH: curr_states}

            topk_ids, topk_log_probs, new_states = sess.run([self.SingleStepDecoder.topk_ids,
                                                             self.SingleStepDecoder.topk_probs,
                                                             self.SingleStepDecoder.new_states], feed_dict=fd)
            #for t in topk_ids:
            #    print(self.word2numb.convert_n2w(t.tolist()))
            #print(topk_log_probs)
            new_states = np.array(new_states)
            #print(new_states)
            #print(new_states.shape)
            #new_states=new_states.reshape((-1, self.numb_dec_layers, 2, self.dec_hidden_units))
            #print(new_states)
            #print(new_states.shape)


            # Extend each hypothesis.
            all_hyps = []
            # The first step takes the best K results from first hyps. Following
            # steps take the best K results from K*K hyps.
            num_beam_source = 1 if steps == 0 else len(hyps)
            for i in range(num_beam_source):
                h, ns = hyps[i], new_states[:,:,i,:]
                for j in range(self._beam_size*2):
                    all_hyps.append(h.Extend(topk_ids[i, j], topk_log_probs[i, j], ns))

            # Filter and collect any hypotheses that have the end token.
            hyps = []
            for h in self._BestHyps(all_hyps):
                if h.latest_token == self._end_token:
                    # Pull the hypothesis off the beam if the end token is reached.
                    results.append(h)
                else:
                    # Otherwise continue to the extend the hypothesis.
                    hyps.append(h)
                if len(hyps) == self._beam_size or len(results) == self._beam_size:
                    break

            steps += 1

        if steps == self._max_steps:
            results.extend(hyps)

        return self._BestHyps(results)

    def _BestHyps(self, hyps, norm_by_len=False):
        """Sort the hyps based on log probs and length.
        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis in reverse log_prob order.
        """
        # This length normalization is only effective for the final results.
        if norm_by_len:
            return sorted(hyps, key=lambda h: h.log_prob/len(h.tokens), reverse=True)
        else:
            return sorted(hyps, key=lambda h: h.log_prob, reverse=True)








class AttnDecoder(object):
    def __init__(self, dec_inputs, encoder_input_len, chan_output, chan_attn_out, helper_prob, embeddings, config):
        self.decoder_inputs = dec_inputs
        self.enc_input_len = encoder_input_len
        self.decoder_lengths = self.enc_input_len + 3
        self.helper_prop = helper_prob
        self.batch_size = config.batch_size


        self.numb_dec_layers = config.numb_dec_layers
        self.dec_hidden_units = config.dec_hidden_units
        self.vocab_size = config.vocab_size
        self.init_state = self.expand_chann_out(chan_output)
        self.chan_attn_out = chan_attn_out

        self.embeddings = embeddings

        self.dec_logits, self.dec_pred = self.build_dec_network


    def expand_chann_out(self, channel_out):
        bits_with_erasure_c = channel_out[0]
        bits_with_erasure_h = channel_out[1]
        init_state = []
        for i in range(self.numb_dec_layers):
            self.state_c_out = tf.layers.dense(bits_with_erasure_c,
                                               self.dec_hidden_units,
                                               activation=None,
                                               name="stateC_out_L{}".format(i))

            self.state_h_out = tf.layers.dense(bits_with_erasure_h,
                                               self.dec_hidden_units,
                                               activation=tf.nn.tanh,
                                               name="stateH_out_L{}".format(i))
            init_state.append(tf.contrib.rnn.LSTMStateTuple(c=self.state_c_out, h=self.state_h_out))
        if self.numb_dec_layers==1:
            return init_state[0]
        else:
            return tuple(init_state)

    @property
    def build_dec_network(self):
        embeds = self.embeddings.get_embeddings(self.decoder_inputs)
        cell = tf.contrib.rnn.LSTMCell(self.dec_hidden_units)
        if self.numb_dec_layers>1:
            cells = [tf.contrib.rnn.LSTMCell(num_units=self.dec_hidden_units, state_is_tuple=True)
                     for _ in range(self.numb_dec_layers)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)
        #attn_data = tf.transpose(self.chan_attn_out, [1, 0, 2])
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.dec_hidden_units, self.chan_attn_out)
        attn_cell = MyAttentionWrapper(cell,
                                       attention_mechanism,
                                       attention_layer_size=self.dec_hidden_units,
                                       cell_input_fn=(lambda inputs, attention: inputs),
                                       initial_cell_state=self.init_state)

        pad_time_slice = tf.zeros([self.batch_size], dtype=tf.int32, name='PAD')

        pad_step_embedded = self.embeddings.get_embeddings(pad_time_slice)

        helper = MyScheduledEmbeddingTrainingHelper(inputs=embeds,
                                                    sequence_length=self.decoder_lengths,
                                                    embedding=self.embeddings.embeddings,
                                                    sampling_probability=self.helper_prop,
                                                    PAD_embbed=pad_step_embedded,
                                                    time_major=False,
                                                    seed=None,
                                                    scheduling_seed=None,
                                                    name=None)

        # helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs=embeds,
        #                                                              sequence_length=self.decoder_lengths,
        #                                                              embedding=self.embeddings.embeddings,
        #                                                              sampling_probability=self.helper_prop,
        #                                                              time_major=True,
        #                                                              seed=None,
        #                                                              scheduling_seed=None,
        #                                                              name=None)

        decoder_cell = tf.contrib.seq2seq.BasicDecoder(attn_cell,
                                                       helper=helper,
                                                       output_layer=layers_core.Dense(self.vocab_size),
                                                       initial_state=attn_cell.zero_state(dtype=tf.float32,
                                                                                          batch_size=self.batch_size))

        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder_cell,
                                                                                               output_time_major=False)

        print("dynamic out", final_outputs.rnn_output)
        decoder_logits = final_outputs.rnn_output

        decoder_prediction = tf.argmax(decoder_logits, 2)
        print(decoder_prediction)

        return (decoder_logits, decoder_prediction)





