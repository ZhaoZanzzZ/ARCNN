import tensorflow as tf
import numpy as np
import os
from PIL import Image
import math

if __name__ == '__main__':
    
    #call GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    sess = tf.Session(config = config)
    
    #generate compresed picture
    img = Image.open(r"/home/zanz/AR-CNN-master/data/kodak/kodim20.png")#Maybe you need to change the directory
    #img.show()
    #print (img.format) 
    #print (img.mode) 
    
    img.save('/home/zanz/AR-CNN-master/data/kodak/kodim20.jpg',quality = 10)
    img_compres = Image.open(r"/home/zanz/AR-CNN-master/data/kodak/kodim20.jpg")
    #img_compres.show()
    col = img_compres.size[0]
    row = img_compres.size[1]


    #generate truths
    img_YUV = img.convert("YCbCr")
    img_Y,img_U,img_V = img_YUV.split()
    img_Y_test=np.asarray(img_Y,np.float32).reshape([1, row, col,1])
    img_Y = np.asarray(img_Y,np.float32)
    img_Y = img_Y.reshape([1, row, col,1]) /255 *100
    

    #generate compres
    img_compres_YUV = img_compres.convert("YCbCr")
    img_compres_Y,img_compres_U,img_compres_V = img_compres_YUV.split()
    img_compres_Y_test=np.asarray(img_compres_Y,np.float32).reshape([1, row, col,1])
    img_compres_Y = np.asarray(img_compres_Y,np.float32)
    img_compres_Y = img_compres_Y.reshape([1, row, col,1]) /255 *100


    #get reconstrction_Y
    reconstruct_Y = np.zeros([1,row, col,1])

    #restore ckpt
    saver=tf.train.import_meta_graph('/home/zanz/AR-CNN-master/model_jpeg/ckpt_quality=10/ckpt_05/model.ckpt.meta')#Maybe you need to change the directory
    saver.restore(sess,tf.train.latest_checkpoint("/home/zanz/AR-CNN-master/model_jpeg/ckpt_quality=10/ckpt_05/"))
    feed_data = {'MyInputCompres:0':img_compres_Y,'MyInputTruth:0':img_Y}
    reconstruct_Y,loss,ori_loss = sess.run(['reconstruction/fan_out/MyOutput:0','loss/loss:0','original_loss/original_loss:0'], feed_dict=feed_data)

     
    reconstruct_Y = reconstruct_Y * 255 /100
    reconstruct_Y = reconstruct_Y.reshape([row,col])
    for i in range(row):
        for j in range(col):
            if reconstruct_Y[i][j] > 255.0:
                reconstruct_Y[i][j] = 255.0
            if reconstruct_Y[i][j] < 0.0:
                reconstruct_Y[i][j] = 0.0

    
    reconstruct_Y = reconstruct_Y.round().astype(np.float32).reshape([1,row,col,1])
    loss =sess.run(tf.reduce_mean(tf.square(reconstruct_Y - img_Y_test)))
    ori_loss = sess.run(tf.reduce_mean(tf.square(img_compres_Y_test - img_Y_test)))
    PSNR_rec = 10.0 * math.log(65025.0 / loss) / math.log(10.0)
    PSNR_jpeg = 10.0 * math.log(65025.0 / ori_loss) / math.log(10.0) 
    print (PSNR_rec,PSNR_jpeg,PSNR_rec - PSNR_jpeg) 


    #get reconstruction picture
    reconstruct_Y = reconstruct_Y.reshape([row,col])
    rec_Y= Image.fromarray(reconstruct_Y.astype(np.uint8))
    img_reconstruct = Image.merge("YCbCr",(rec_Y,img_compres_U,img_compres_V))
    img_rec = img_reconstruct.convert("RGB")
    img_rec.save('/home/zanz/AR-CNN-master/data/kodak/img_reconstruct.png')
    #print(img_rec.format)
    #print(img_rec.mode)