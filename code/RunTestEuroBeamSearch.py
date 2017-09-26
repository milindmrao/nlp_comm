import os
import tensorflow as tf
import numpy as np
from SentenceBatchGenerator import SentenceBatchGenerator, Word2Numb
from EncDecChanModels import Config
from SentenceEncChanDecNet import generate_tb_filename, BeamSearchEncChanDecNet


def beam_test_on_euro_testset(sess, beamNN, test_data, test_results_path):
    # =============================  Validate on Test Data ===============================
    test_data.prepare_batch_queues(randomize=False)
    batch = test_data.get_next_batch(randomize=False)
    numb_errors = 0
    numb_words = 0
    with open(test_results_path, 'w', newline='') as file:
        while batch != None:
            batch = batch[0]
            #print(batch)
            chan_out = beamNN.encode_Tx_sentence(sess, batch, keep_rate=1.0)
            bestseq, bestseq_prob, all_beams = beamNN.dec_Rx_bits(sess, chan_out)

            pred = all_beams[0].tokens
            pred = pred[1:]
            diff = [int(pred[i] != batch[i]) for i in range(min(len(pred), len(batch)))]
            numb_words += max(len(pred), len(batch))
            curr_errors = sum(diff) + abs(len(pred)-len(batch))
            numb_errors += curr_errors
            tx = " ".join(beamNN.word2numb.convert_n2w(batch))
            rx = " ".join(beamNN.word2numb.convert_n2w(pred))
            file.write('TX: {}\n'.format(tx))
            file.write('RX: {}\n'.format(rx))
            if curr_errors > 0:
                accuracy = numb_errors / numb_words
                print('  ==================================== WER {}'.format(accuracy))
                print('TX: {}'.format(tx))
                print('RX: {}'.format(rx))
            #print(numb_words)
            #print(numb_errors)

            # acc_list.append(accu_)
            #
            # for i, (inp, pred) in enumerate(zip(fd[sysNN.enc_inputs], predict_)):
            #     tx = " ".join(sysNN.word2numb.convert_n2w(inp))
            #     rx = " ".join(sysNN.word2numb.convert_n2w(pred))
            #     if i < 3:
            #         print('  sample {}:'.format(i + 1))
            #         print('TX: {}'.format(tx))
            #         print('RX: {}'.format(rx))
            #     file.write('TX: {}\n'.format(tx))
            #     file.write('RX: {}\n'.format(rx))

            batch = test_data.get_next_batch(randomize=False)
        accuracy = 1 - numb_errors/numb_words
        print("Average Accuracy: ", accuracy)
        file.write("Average Accuracy: {}\n".format(accuracy))
    return


if __name__ == '__main__':
    chan_params = {"type": "erasure", "keep_prob": 1.0}
    beam_size = 10

    parent_dir, _ = os.path.split(os.getcwd())
    # parent_dir = os.getcwd()
    print(parent_dir)


    print('Init and Loading Data...')
    config = Config(None,chan_params, lr=0.001, peephole=True, batch_size=1)
    # config = Config(None, chan_params)
    resultsPath = generate_tb_filename(config)
    resultsPath += "-BeamSearch-" + str(beam_size) + ".txt"
    #modelPath = "Chan-erasure0.9-lr-0.001-txbits-500-voc-19158-embed-50-lstm-256-peep-True-epochs-30-bs-128-SOSdata-EuroSentence-Binarizer-HardAttn-0EncLayers-1-DecLayers-1"
    modelPath = "Chan-erasure0.9-lr-0.001-txbits-600-voc-19158-embed-200-lstm-256-peep-True-epochs-30-bs-128-SOSdata-EuroSentence-Binarizer-HardAttn-0EncLayers-2-DecLayers-2-VarProbCorrect"

    model_save_path = os.path.join(parent_dir, 'trained_models', 'Vocab-19158', modelPath)
    test_results_path = os.path.join(parent_dir, 'test_results', resultsPath)
    config.model_save_path=model_save_path
    word2numb = Word2Numb(config.w2n_path)
    train_sentence_gen = None

    test_sentences = SentenceBatchGenerator(config.testdata_path,
                                            word2numb,
                                            config.batch_size,
                                            config.length_from,
                                            config.length_to,
                                            config.bin_len,
                                            config.UNK)


    print('Done!')
    print('Building Network...')
    beam_sys = BeamSearchEncChanDecNet(config, word2numb, beam_size=beam_size)
    print('Done!')

    print('Start session...')
    with tf.Session() as sess:
        beam_sys.load_enc_dec_weights(sess, model_save_path)
        beam_test_on_euro_testset(sess, beam_sys, test_sentences, test_results_path)
