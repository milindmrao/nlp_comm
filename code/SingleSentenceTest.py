import os
import zlib
import reedsolo
import sys
import tensorflow as tf
import numpy as np
from SentenceBatchGenerator import SentenceBatchGenerator, Word2Numb
from EncDecChanModels import Config
import nltk
from SentenceEncChanDecNet import BeamSearchEncChanDecNet


def compress_sentence(sentence, **kwargs):
    """ Accepts a batch of sentences. Perhaps 32. Compresses it using a universal
    compressor such as gzip. Then expands it based on the parameters.
    Inputs:
        batch (list of sentences): list of sentences
        Optional params:
            'bit_drop_rate'(float or list-of-float): Default 0.01

        Returns:
            encoding (bit-stream)
            compression_ratio
            bits_per_sentence
    """
    #n_sens = len(sentence)
    #sentences_all = str.encode(' '.join(sentence))
    sentence = str.encode(sentence)
    uncomp_len = sys.getsizeof(sentence)
    print("uncompressd size (bits)", uncomp_len*8)
    compressed = zlib.compress(sentence, 9)
    comp_len = sys.getsizeof(compressed)
    print("compressd size (bits)", comp_len * 8)


    # bdr = int(2 * np.ceil(
    #     255 * kwargs.get('bit_drop_rate', 0.10)))  # This ensures that 0.01 * 255 (message size) gets reconstructed
    # channel_coder = reedsolo.RSCodec(bdr)
    # encoded = channel_coder.encode(compressed)
    #
    # comp_ratio = sys.getsizeof(encoded) / sys.getsizeof(sentences_all)
    # bits_per_sentence = sys.getsizeof(encoded) / n_sens
    #
    # return (encoded, comp_ratio, bits_per_sentence)

if __name__ == '__main__':
    chan_params = {"type": "erasure", "keep_prob": 0.9}

    parent_dir, _ = os.path.split(os.getcwd())
    # parent_dir = os.getcwd()
    print(parent_dir)

    print('Init and Loading Data...')
    config = Config(None,chan_params, lr=0.001, peephole=True, batch_size=1)
    word2numb = Word2Numb(config.w2n_path)

    #modelPath = "Chan-erasure0.9-lr-0.001-txbits-500-voc-19158-embed-50-lstm-256-peep-True-epochs-30-bs-128-SOSdata-EuroSentence-Binarizer-HardAttn-0EncLayers-1-DecLayers-1"
    modelPath = "Chan-erasure0.9-lr-0.001-txbits-600-voc-19158-embed-200-lstm-256-peep-True-epochs-30-bs-128-SOSdata-EuroSentence-Binarizer-HardAttn-0EncLayers-2-DecLayers-2-VarProbCorrect"
    model_save_path = os.path.join(parent_dir, 'trained_models', 'Vocab-19158', modelPath)


    # test_sentence = "The cat sat on the floor."
    # test_sentence = "this debate is a big sham."
    # test_sentence = "china is still terrorised by the chinese communist party after 61 years of power."
    #test_sentence = "according to the rules of procedure , the same member may not ask further questions."
    #test_sentence = "as you all know , member states have the opportunity to levy higher infrastructure charges during peak periods."
    #test_sentence = "but multinational oil companies, including european ones such as total, continue to support the military regime."
    #test_sentence = "the high number of injuries arising from accidents or violence in the member states remains cause for concern."
    #test_sentence = "european union progress in 1997"
    #test_sentence = "female human rights defenders face greater difficulties in carrying out their work."
    #test_sentence = "firstly , i would like to offer my sincere thanks to the commissioner for not categorically arguing against the concept of minimum requirements for the safety network."
    #test_sentence = "given how complicated these two directives are and how quickly the market is developing, the committee proposes a review of the rules within the next three years."
    test_sentence = "given how complicated these two directives are and how quickly the market is developing, the committee proposes a review."
    test_sentence = "I am running around."
    #test_sentence = "our leaders are letting us down."
    #print(len(test_sentence))
    #compress_sentence(test_sentence)
    word_token = nltk.tokenize.WordPunctTokenizer()
    words = word_token.tokenize(test_sentence)
    words = [w.lower() for w in words]
    tokens = word2numb.convert_w2n(words)
    print(len(tokens))



    beam_sys = BeamSearchEncChanDecNet(config, word2numb, beam_size=1)

    print('Start session...')
    with tf.Session() as sess:
        beam_sys.load_enc_dec_weights(sess, model_save_path)
        chan_out = beam_sys.encode_Tx_sentence(sess,tokens,keep_rate=0.9)
        bestseq, bestseq_prob, all_beams = beam_sys.dec_Rx_bits(sess,chan_out)
        for B in all_beams:
            print("LogProb:", B.log_prob, "Sentence:",  " ".join(word2numb.convert_n2w(B.tokens)))

        bestseq, bestseq_prob, all_beams = beam_sys.dec_Rx_bits(sess,chan_out, beam_size=10)
        for B in all_beams:
            print("LogProb:", B.log_prob, "Sentence:",  " ".join(word2numb.convert_n2w(B.tokens)))