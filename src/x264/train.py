import tensorflow as tf
import numpy as np
import os
import argparse
import math
from models import *
from utils import *
from BSDS500 import *

def run(conf, data):
    
    sess = tf.Session(config=tf.ConfigProto())
  
    print ('Model Defining...')
    truths = tf.placeholder(tf.float32, [None, None, None, conf.channel],name = 'MyInputTruth')
    print (truths)
    compres = tf.placeholder(tf.float32, [None, None, None, conf.channel],name = 'MyInputCompres')
    print (compres)
    model = ARCNN(conf, truths, compres)

    optimizer = tf.train.AdamOptimizer(1e-3).minimize(model.loss)
    sess.run(tf.initialize_all_variables())


    saver = tf.train.Saver()

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(conf.summary_path, sess.graph)
    
    if os.path.exists(conf.ckpt_path):
        ckpt = tf.train.get_checkpoint_state(conf.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess, ckpt.model_checkpoint_path)
            print ('Variables Restored.')
        else:
            sess.run(tf.global_variables_initializer())
            print ('Variables Initialized.')
    else:
        sess.run(tf.global_variables_initializer())
        print ('Variables Initialized.')  

    
    print ('Training Start.')
    for i in range(conf.epochs):

        for j in range(conf.num_batches):
            batch_truth, batch_compres = data.train.next_batch()
            data_dict = {compres:batch_compres, truths:batch_truth}
            _, cost, ori_cost, summary = sess.run([optimizer, model.loss, model.original_loss, merged], feed_dict=data_dict)
        
            if (j + 1) % 100 == 0:
                print ('%d:%f%%'%(j,((ori_cost - cost)/ ori_cost * 100)))
            
                    
        writer.add_summary(summary, i)

        PSNR = 10.0 * math.log(10000.0 / cost) / math.log(10.0)
        print ('Epoch: %d, Loss: %f, Original Loss: %f, PSNR: %f' % (i, cost, ori_cost, PSNR))
        print ('enhancement: %f%% ' %(((ori_cost - cost)/ ori_cost * 100)))
        saver.save(sess, conf.ckpt_path + '/model.ckpt')
        print ('saved ckpt!')
        
        print ('Validating Start.')
        # num_val_epochs = conf.num_val / conf.test_size + 1
        cost_sum = 0.0
        ori_cost_sum = 0.0
        valid_batches = 100
        for k in range(valid_batches):
            batch_truth, batch_compres = data.test.next_batch()
            data_dict = {compres:batch_compres, truths:batch_truth}

            cost, ori_cost = sess.run([model.loss, model.original_loss], feed_dict=data_dict)
            cost_sum = cost_sum + cost
            ori_cost_sum = ori_cost_sum + ori_cost
            
        cost = cost_sum / valid_batches
        ori_cost = ori_cost_sum / valid_batches
        PSNR = 10.0 * math.log(10000.0 / cost) / math.log(10.0)
        print ('Average Loss: %f, Original Loss: %f, PSNR: %f' % (cost, ori_cost, PSNR))
        print ('enhancement: %f%% ' %(((ori_cost - cost)/ ori_cost * 100)))
 
        print ('Validating completed.')

    model.save(sess)
    print ('Training completed.')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=128)
    parser.add_argument('--quality', type=int, default=32)
    
    parser.add_argument('--data_path', type=str, default='../ProcessedData_x264/train')
    parser.add_argument('--summary_path', type=str, default='../encoder/model_x264/logs')
    parser.add_argument('--ckpt_path', type=str, default='../encoder/model_x264/ckpts')
    parser.add_argument('--param_path', type=str, default='../encoder/model_x264/params')
    conf = parser.parse_args()
  
    data = BSDS500(conf.data_path, conf.batch_size, conf.quality)
    conf.img_height = 32
    conf.img_width = 32
    conf.channel = 1
    conf.valid_height = 20
    conf.valid_width = 20
    conf.num_train = 552000
    conf.num_val = 138000
    conf.num_batches = 3900
    # conf.num_batches = num_train / conf.batch_size + 1
    conf.phase = 'train'

    conf = makepaths(conf) 

    os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    run(conf, data)

