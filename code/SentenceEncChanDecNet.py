import time
import os
import numpy as np
import tensorflow as tf
from EncDecChanModels import Embedding, SimpleEncoder, SimpleChannel, AttnDecoder, SimpleDecoder, BeamSearch

def generate_tb_filename(config):
    tb_name = ""
    if config.chan_params["type"]=="erasure":
        tb_name = "Chan-" + config.chan_params["type"] + str(config.chan_params["keep_prob"])
    elif config.chan_params["type"]=="none":
        tb_name = "Chan-" + config.chan_params["type"]

    tb_name += "-lr-" + str(config.lr)+"-txbits-"+str(config.numb_tx_bits)+"-voc-"+str(config.vocab_size)
    #tb_name += "-lr-sqrt0.01" + "-txbits-" + str(config.numb_tx_bits) + "-voc-" + str(config.vocab_size)
    tb_name += "-embed-" + str(config.embedding_size)+"-lstm-"+str(config.enc_hidden_units)
    tb_name += "-peep-" + str(config.peephole)
    tb_name += "-epochs-" + str(config.epochs) + "-bs-" + str(config.batch_size)
    if config.SOS == None:
        tb_name += "-NoSOS"
    else:
        tb_name += "-SOS"

    tb_name += "data-EuroSentence"
    tb_name += "-Binarizer"
    tb_name += "-HardAttn-"+str(config.numb_attn_bits)
    tb_name += "EncLayers-"+str(config.numb_enc_layers)
    tb_name += "-DecLayers-"+str(config.numb_dec_layers)
    tb_name += "-VarProbCorrect"
    return tb_name


class SimpleSystem(object):
    def __init__(self, config, train_data, test_data, word2numb):
        self.config = config
        self.training_counter = 1
        self.test_counter = 1
        self.train_data = train_data
        self.test_data = test_data
        self.word2numb = word2numb

        # ==== reset graph ====
        tf.reset_default_graph()


        # ==== Placeholders ====
        self.isTrain = tf.placeholder(tf.bool, shape=(), name='isTrain')
        self.enc_inputs = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='encoder_inputs')
        self.enc_inputs_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        self.dec_inputs = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='decoder_targets')
        self.dec_targets = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int64, name='decoder_targets')
        self.helper_prob = tf.placeholder(shape=[], dtype=tf.float32, name='helper_prob')
        self.keep_rate = tf.placeholder(shape=[], dtype=tf.float32, name='keep_rate')
        self.lr = tf.placeholder(shape=[], dtype=tf.float32, name='lr')
        #self.prob_dec_corr_input = tf.placeholder(shape=[], dtype=tf.float32, name='prob_corr_input')


        # ==== Building neural network graph ====
        self.embeddings = Embedding(self.config)
        self.encoder = SimpleEncoder(self.enc_inputs, self.enc_inputs_len, self.embeddings, self.config)
        self.channel = SimpleChannel(self.encoder.enc_outputs,
                                     self.encoder.enc_state_c,
                                     self.encoder.enc_state_h,
                                     self.isTrain,
                                     self.keep_rate,
                                     self.config)

        if (self.config.numb_attn_bits>0):
            self.decoder = AttnDecoder(self.dec_inputs,
                                       self.enc_inputs_len,
                                       self.channel.channel_out,
                                       self.channel.attn_out,
                                       self.helper_prob,
                                       self.embeddings,
                                       self.config)
        else:
            self.decoder = SimpleDecoder(self.enc_inputs,
                                         self.enc_inputs_len,
                                         self.channel.channel_out,
                                         self.embeddings,
                                         self.dec_targets,
                                         self.helper_prob,
                                         self.config)

        # ==== define loss and training op and accuracy ====
        self.loss, self.train_op = self.define_loss()
        self.accuracy = self.define_accuracy()

        # ==== set up training/updating procedure ====
        #self.saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=0.25)
        self.saver = tf.train.Saver(max_to_keep=0)

        tf.summary.scalar("CrossEntLoss", self.loss)
        tf.summary.histogram("enc_state_c", self.encoder.enc_state_c)
        tf.summary.histogram("enc_state_h", self.encoder.enc_state_h)
        if self.config.chan_params["type"] != "none":
            tf.summary.histogram("state_c_reduced", self.channel.state_c_reduc)
            tf.summary.histogram("state_h_reduced", self.channel.state_h_reduc)
            tf.summary.histogram("state_c_bits", self.channel.state_c_bits)
            tf.summary.histogram("state_h_bits", self.channel.state_h_bits)
        tf.summary.histogram("chan_state_c", self.channel.state_c_out)
        tf.summary.histogram("chan_state_h", self.channel.state_h_out)
        self.tb_summary = tf.summary.merge_all()
        self.tb_val_summ = tf.summary.scalar("Validation_Accuracy", self.accuracy)



    def load_trained_model(self, sess, trained_model_path):
        """
        Loads a trained model from what was saved. Insert the trained model path
        Inputs:
            trained_model_path (str-path) - Path for storing the training model
        Returns:
            Nothing. Prints if model parameters are loaded or not.
        """
        trained_model_folder = os.path.split(trained_model_path)[0]
        ckpt = tf.train.get_checkpoint_state(trained_model_folder)
        v2_path = os.path.join(trained_model_folder, os.path.split(ckpt.model_checkpoint_path)[1] + ".index")
        norm_ckpt_path = os.path.join(trained_model_folder, os.path.split(ckpt.model_checkpoint_path)[1])
        if ckpt and (tf.gfile.Exists(norm_ckpt_path) or
                         tf.gfile.Exists(v2_path)):
            print("Reading model parameters from %s" % norm_ckpt_path)
            self.saver.restore(sess, norm_ckpt_path)
        else:
            print('Error reading weights')

    def define_accuracy(self):
        eq_indicator = tf.cast(tf.equal(self.decoder.dec_pred, self.dec_targets), dtype=tf.float32)
        return tf.reduce_mean(eq_indicator)


    def define_loss(self):
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.dec_targets, depth=self.config.vocab_size, dtype=tf.float32),
            logits=self.decoder.dec_logits,
        )
        # loss function
        loss = tf.reduce_mean(stepwise_cross_entropy)
        # train it
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        return loss, train_op

    def batch_to_feed(self, inputs, max_sequence_length=None, time_major=False):
        """
        Args:
            inputs:
                list of sentences (integer lists)
            max_sequence_length:
                integer specifying how large should `max_time` dimension be.
                If None, maximum sequence length would be used

        Outputs:
            batch_out: zero padded batch
            sequence_lengths: sentence len
        """
        sequence_lengths = [len(seq) for seq in inputs]
        batch_size = len(inputs)

        if max_sequence_length is None:
            max_sequence_length = max(sequence_lengths)

        inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

        for i, seq in enumerate(inputs):
            for j, element in enumerate(seq):
                inputs_batch_major[i, j] = element

        if time_major:
            # [batch_size, max_time] -> [max_time, batch_size]
            batch_out = inputs_batch_major.swapaxes(0, 1)
        else:
            batch_out = inputs_batch_major

        return batch_out, sequence_lengths

    def next_feed(self,batch,lr, help_prob = 1.0, isTrain=True, keep_rate=1.0):
        #batch = next(batches)
        encoder_inputs_, encoder_input_lengths_ = self.batch_to_feed(
            [(sequence) + [self.config.EOS] for sequence in batch])
        decoder_targets_, _ = self.batch_to_feed(
            [(sequence) + [self.config.EOS] + [self.config.PAD] * 3 for sequence in batch])
        decoder_inputs_, _ = self.batch_to_feed(
            [[self.config.SOS] + (sequence) + [self.config.EOS] + [self.config.PAD] * 2 for sequence in batch])

        #print(encoder_inputs_)
        #print(decoder_inputs_)
        #print(decoder_targets_)
        return {self.isTrain: isTrain,
                self.enc_inputs: encoder_inputs_,
                self.enc_inputs_len: encoder_input_lengths_,
                self.dec_inputs: decoder_inputs_,
                self.dec_targets: decoder_targets_,
                self.helper_prob: help_prob,
                self.keep_rate:keep_rate,
                self.lr: lr}




    def train(self, sess, tb_writer, grammar=None):

        # batches = SimpleDataGenerator.random_sequences(length_from=3, length_to=18,
        #                                                vocab_lower=2, vocab_upper=self.config.vocab_size - 1,
        #                                                batch_size=self.config.batch_size)
        params = tf.trainable_variables()
        num_params = sum(
            map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        print('Total model parameters: ', num_params)
        help_prob = 1.0
        try:
            self.training_counter = 1
            self.test_counter = 1
            for i in range(self.config.epochs):
                if i > 10:
                    help_prob = max(0.0, help_prob - 0.05)



                # =============================  Train on Training Data ===============================
                self.train_data.prepare_batch_queues()
                batch = self.train_data.get_next_batch()
                print_every = 100
                tic = time.time()
                while batch != None:

                    # if batch_idx <= self.config.batches_in_epoch*5:
                    #      lr = 0.001
                    # elif batch_idx>self.config.batches_in_epoch*5 and batch_idx <= self.config.batches_in_epoch*15:
                    #     # lr = self.config.lr/np.log2(self.tb_itr_counter)
                    #     lr = 0.0001
                    # else:
                    #     lr = 0.00005
                    lr = self.config.lr

                    fd = self.next_feed(batch,lr, help_prob=help_prob,
                                        keep_rate=self.config.chan_params["keep_prob"])
                    _, loss, tb_summ = sess.run([self.train_op, self.loss, self.tb_summary], fd)

                    tb_writer.add_summary(tb_summ, self.training_counter)
                    self.training_counter += 1

                    batch = self.train_data.get_next_batch()

                    if self.training_counter % print_every == 0 or batch == None:
                        toc = time.time()
                        print("-- Epoch: ", i+1,
                              "Numb Batches: ", print_every,
                              "Training Time: ", toc - tic,
                              "Training Loss: ", loss)
                        if batch != None:
                            tic = time.time()


                # =============================  Validate on Test Data ===============================
                self.test_data.prepare_batch_queues()
                batch = self.test_data.get_next_batch()
                acc_list = []
                tic = time.time()
                while batch != None:

                    lr = self.config.lr

                    fd = self.next_feed(batch, lr, isTrain=False, help_prob=0.0,
                                        keep_rate=self.config.chan_params["keep_prob"])
                    predict_, accu_, tb_summ = sess.run([self.decoder.dec_pred, self.accuracy, self.tb_val_summ], fd)

                    tb_writer.add_summary(tb_summ, self.test_counter)

                    acc_list.append(accu_)

                    self.test_counter += 1

                    batch = self.test_data.get_next_batch()
                    if batch == None:
                        toc = time.time()
                        print("-- Test Time: ", toc - tic,
                              "Average Accuracy: ", np.average(acc_list))

                        for j, (inp, pred) in enumerate(zip(fd[self.enc_inputs], predict_)):
                            if j >= 10:
                                break
                            tx = " ".join(self.word2numb.convert_n2w(inp))
                            rx = " ".join(self.word2numb.convert_n2w(pred))
                            print('  sample {}:'.format(j + 1))
                            print('    input     > {}'.format(tx))
                            print('    predicted > {}'.format(rx))

                #fileName = self.config.model_save_path
                self.saver.save(sess, self.config.model_save_path, global_step=i)
                print("Model saved in file: %s" % self.config.model_save_path)

        except KeyboardInterrupt:
            print('training interrupted')

        self.saver.save(sess, self.config.model_save_path)
        print("Model saved in file: %s" % self.config.model_save_path)

        return


class BeamSearchEncChanDecNet(object):
    def __init__(self, config, word2numb, beam_size=10):
        self.config = config
        self.word2numb = word2numb
        self.beam_size = 10

        # ==== reset graph ====
        tf.reset_default_graph()

        # ==== Placeholders ====
        self.isTrain = tf.placeholder(tf.bool, shape=(), name='isTrain')
        self.sentence = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.sentence_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        self.dec_input = tf.placeholder(shape=(None,), dtype=tf.int32, name='dec_input')
        self.dec_state = tf.placeholder(shape=(config.numb_dec_layers, 2, None, config.dec_hidden_units),
                                   dtype=tf.float32, name='dec_state')
        self.chan_out_PH = tf.placeholder(shape=(2, None, config.numb_tx_bits / 2),
                                     dtype=tf.float32, name='chan_out')
        self.keep_rate = tf.placeholder(shape=[], dtype=tf.float32, name='keep_rate')

        # ==== Building neural network graph ====
        self.embeddings = Embedding(config)
        self.encoder = SimpleEncoder(self.sentence,
                                     self.sentence_len,
                                     self.embeddings,
                                     self.config)

        self.channel = SimpleChannel(self.encoder.enc_outputs,
                                     self.encoder.enc_state_c,
                                     self.encoder.enc_state_h,
                                     self.isTrain,
                                     self.keep_rate,
                                     self.config)

        self.beam_search_dec = BeamSearch(self.embeddings,
                                          self.dec_input,
                                          self.dec_state,
                                          self.chan_out_PH,
                                          self.word2numb,
                                          beam_size=beam_size,
                                          config=config)

        self.saver = tf.train.Saver()

    def load_enc_dec_weights(self, sess, model_filepath):
        sess.run(tf.global_variables_initializer())
        self.saver.restore(sess, model_filepath)


    def encode_Tx_sentence(self, sess, num_tokens, keep_rate=1.0):
        num_tokens.append(self.config.EOS)
        fd = {self.isTrain:False,
              self.sentence:[num_tokens],
              self.sentence_len:[len(num_tokens)],
              self.keep_rate:keep_rate}
        chan_out = sess.run([self.channel.channel_out],fd)
        chan_out = np.array(chan_out)
        return chan_out[0]


    def dec_Rx_bits(self, sess, chan_output, beam_size=None):
        beams = self.beam_search_dec.BeamSearch(sess, chan_output, beam_size=beam_size)
        bestseq = " ".join(self.word2numb.convert_n2w(beams[0].tokens))
        bestseq_prob = beams[0].log_prob
        return bestseq, bestseq_prob, beams