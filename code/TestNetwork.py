import os
import time
import tensorflow as tf
from SimpleNetwork import EncChanDecNN
from nnmodels import Config

if __name__ == '__main__':
    tf.reset_default_graph()  # Clears the default graph stack and resets the global default graph.
    ##sess = tf.InteractiveSession()  # initializes a tensorflow session

    parent_dir, _ = os.path.split(os.getcwd())
    parent_dir, _ = os.path.split(parent_dir)
    emb_path = os.path.join(parent_dir, 'data', '50_embed_large.pickle')
    w2n_path = os.path.join(parent_dir, 'data', 'w2n_n2w.pickle')
    # train_data_path = os.path.join(parent_dir, 'data', 'training_euro.pickle')
    train_data_path = os.path.join(parent_dir, 'data', 'training_euro.pickle')
    valid_data_path = os.path.join(parent_dir, 'data', 'training_euro.pickle')
    curr_time = str(time.time())
    trained_model_path = os.path.join(parent_dir, 'trained_models', '1496366210.406971model.weights')

    print('Building network...')
    config = Config(emb_path, w2n_path, train_data_path, valid_data_path, trained_model_path, training=False)
    config.batch_size = 1
    print(config.batch_size)
    comNN = EncChanDecNN(config)
    print('Done!')

    #train_data, _ = comNN.load_data()
    #print(len(train_data))
    print('Start session...')
    with tf.Session() as sess:
        # tfk.backend.set_session(sess)
        # tfk.backend.set_learning_phase(1)
        sess.run(tf.global_variables_initializer())
        comNN.load_trained_model(sess,trained_model_path)
        comNN.test_network_performance(sess)