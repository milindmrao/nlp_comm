import numpy as np
import pickle
import random
from collections import deque


class Word2Numb(object):
    def __init__(self, w2n_path, UNK_ID=3):
        # ==== load num2word and word2num =======
        with open(w2n_path, 'rb') as fop:
            [self.w2n, self.n2w] = pickle.load(fop)
        print(len(self.w2n))
        self.UNK_ID = UNK_ID

    def convert_w2n(self, sentence):
        return [self.w2n.get(x, self.UNK_ID) for x in sentence]

    def convert_n2w(self, numbs):
        return [self.n2w.get(x, "<>") for x in numbs]

class SentenceBatchGenerator (object):
    def __init__(self, corp_path,
                 word2numb,
                 batch_size,
                 min_len,
                 max_len,
                 diff,
                 UNK_ID = 3,
                 first_x_sent=0):


        self.corp_path = corp_path
        self.word2numb = word2numb
        self.batch_size = batch_size
        self.min_len = min_len
        self.max_len = max_len
        self.diff = diff
        self.UNK_ID = UNK_ID
        #self.first_x_sent = first_x_sent

        self.sent_as_num = self.load_sentences_as_num(first_x_sent)
        print("Length of Training Data", len(self.sent_as_num))

        self.numb_queues = 0
        self.batch_queues = []




    def load_sentences_as_num(self, first_x_sent):
        """ Function reads a file with tokenized strings (i.e., sentences broken into words).
        It converts this to a sequence of token numbers and returns
        Inputs:
            file_path : path of the file to read.
            word2num : from the vocabulary

        Returns:
            List of lists. Padding is not applied here.

        Point to note: <unk> is 0.
        """
        sentences = []
        with open(self.corp_path, 'rb') as fop:
            sentence_raw = pickle.load(fop)

        for i, single_sentence in enumerate(sentence_raw):
            if first_x_sent > 0 and i>=first_x_sent:
                break
            tokens = self.word2numb.convert_w2n(single_sentence)
            if sum([x == self.UNK_ID for x in tokens]) / len(tokens) < 0.2:
                sentences += [tokens]

        return sentences


    def init_batch_queues(self):
        self.batch_queues = []
        self.numb_queues = int(np.ceil((self.max_len - self.min_len) / self.diff))
        for i in range(self.numb_queues):
            new_queue = deque()
            self.batch_queues.append(new_queue)

        #print(self.batch_queues)



    def prepare_batch_queues(self, max_numb_batches=0, randomize=True):
        self.init_batch_queues()
        if randomize:
            random.shuffle(self.sent_as_num)

        for i, sentence in enumerate(self.sent_as_num):
            if max_numb_batches > 0 and i >= max_numb_batches:
                break
            sent_len = len(sentence)
            idx = int(np.floor((sent_len - self.min_len) / self.diff))
            if sent_len == self.max_len:
                idx = self.numb_queues - 1

            self.batch_queues[idx].append(sentence)
        #print(self.batch_queues)


    def get_next_batch(self,randomize=True):
        batch = None
        find_rand_Q = True
        while find_rand_Q and len(self.batch_queues):
            queue_idx = 0
            if randomize:
                queue_idx = random.randrange(0, self.numb_queues)
            if len(self.batch_queues[queue_idx]) >= self.batch_size:
                find_rand_Q = False
                batch = [self.batch_queues[queue_idx].pop() for _ in range(self.batch_size)]
            else:
                del self.batch_queues[queue_idx]
                self.numb_queues -= 1

        return batch




if __name__ == "__main__":
    w2n_path = "../../data/w2n_n2w_TopEuro.pickle"
    traindata_path = "../../data/training_euro_wordlist20.pickle"
    testdata_path = "../../data/testing_euro_wordlist20.pickle"
    w2n = Word2Numb(w2n_path)
    generator = SentenceBatchGenerator(testdata_path, w2n, 3, 4, 20, 4,first_x_sent=20)
    generator.prepare_batch_queues()
    for i in range(generator.numb_queues):
        print(len(generator.batch_queues[i]))

    batch = generator.get_next_batch()
    print(batch)
    while batch != None:
        batch = generator.get_next_batch()
        print(np.asarray(batch))
    pass