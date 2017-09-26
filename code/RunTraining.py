import os
import tensorflow as tf
from SentenceBatchGenerator import SentenceBatchGenerator, Word2Numb
from EncDecChanModels import Config
from SentenceEncChanDecNet import generate_tb_filename, SimpleSystem


if __name__ == '__main__':
    chan_params = {"type": "erasure", "keep_prob": 0.9}

    print("hello!")
    parent_dir, _ = os.path.split(os.getcwd())
    #parent_dir = os.getcwd()
    print(parent_dir)

    print('Init and Loading Data...')
    config = Config(None,chan_params,lr=0.001)
    #config = Config(None, chan_params)
    fileName = generate_tb_filename(config)
    model_save_path = os.path.join(parent_dir, 'trained_models', 'Vocab-{}'.format(config.vocab_size), fileName)
    config.model_save_path=model_save_path
    word2numb = Word2Numb(config.w2n_path)
    train_sentence_gen = SentenceBatchGenerator(config.traindata_path,
                                                word2numb,
                                                config.batch_size,
                                                config.length_from,
                                                config.length_to,
                                                config.bin_len,
                                                config.UNK)

    test_sentences = SentenceBatchGenerator(config.testdata_path,
                                            word2numb,
                                            config.batch_size,
                                            config.length_from,
                                            config.length_to,
                                            config.bin_len,
                                            config.UNK,
                                            first_x_sent=7000)
    print('Done!')
    print('Building Network...')
    simp_sys = SimpleSystem(config, train_sentence_gen, test_sentences, word2numb)
    print('Done!')





    summ_path = os.path.join(parent_dir, 'tensorboard', 'Vocab-{}'.format(config.vocab_size), fileName)


    print('Start training...')
    with tf.Session() as sess:
        # tfk.backend.set_session(sess)
        # tfk.backend.set_learning_phase(1)
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(summ_path)
        writer.add_graph(sess.graph)
        simp_sys.train(sess, writer)
    print('Finished training!')