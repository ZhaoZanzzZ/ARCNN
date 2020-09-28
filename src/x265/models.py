import tensorflow as tf
from layers import *
import os

class ARCNN(object):
    def __init__(self, conf, truths, compres):
        self.conf = conf
        self.truths = truths
        self.compres = compres

        # layer1: feature extraction
        self.W_1, self.b_1, self.F_1 = conv_layer(conf, 'feature_extraction', self.compres, 3, 32, True)
        #conv_layer(conf, layer_name, fan_in, kernel_size, f_map, activation)
        # layer2: feature enhancement
        self.W_2, self.b_2, self.F_2 = conv_layer(conf, 'feature_enhancement', self.F_1, 3, 32, True)

        # layer3: mapping_1
        self.W_3, self.b_3, self.F_3 = conv_layer(conf, 'mapping_1', self.F_2, 3, 32, True)
        
        # layer4: mapping_2
        self.W_4, self.b_4, self.F_4 = conv_layer(conf, 'mapping_2', self.F_3, 3, 32, True)
        
        # layer5: mapping_3
        self.W_5, self.b_5, self.F_5 = conv_layer(conf, 'mapping_3', self.F_4, 3, 32, True)

        # layer6: reconstruction
        self.W_6, self.b_6, self.F_6 = conv_layer(conf, 'reconstruction', self.F_5, 3, conf.channel, False)


        mid_compres = self.compres
        mid_reconstruct = self.F_6
        mid_truths = self.truths
        
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(mid_reconstruct - mid_truths),name='loss')
            variable_summaries(self.loss)
            
        with tf.name_scope('original_loss'):
            self.original_loss = tf.reduce_mean(tf.square(mid_compres - mid_truths),name='original_loss')
            variable_summaries(self.original_loss)

    
    def save(self, sess):
        # layer 1
        filename = os.path.join(self.conf.param_path, 'feature_extraction.npz')
        W = sess.run(self.W_1)
        b = sess.run(self.b_1)
        np.savez(filename, W=W, b=b)

        # layer 2
        filename = os.path.join(self.conf.param_path, 'feature_enhancement.npz')
        W = sess.run(self.W_2)
        b = sess.run(self.b_2)
        np.savez(filename, W=W, b=b)

        # layer 3
        filename = os.path.join(self.conf.param_path, 'mapping_1.npz')
        W = sess.run(self.W_3)
        b = sess.run(self.b_3)
        np.savez(filename, W=W, b=b)

        # layer 4
        filename = os.path.join(self.conf.param_path, 'mapping_2.npz')
        W = sess.run(self.W_4)
        b = sess.run(self.b_4)
        np.savez(filename, W=W, b=b)
        
        # layer 5
        filename = os.path.join(self.conf.param_path, 'mapping_3.npz')
        W = sess.run(self.W_5)
        b = sess.run(self.b_5)
        np.savez(filename, W=W, b=b)

        # layer 6
        filename = os.path.join(self.conf.param_path, 'reconstruction.npz')
        W = sess.run(self.W_6)
        b = sess.run(self.b_6)
        np.savez(filename, W=W, b=b)
