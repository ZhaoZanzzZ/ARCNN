import tensorflow as tf
import numpy as np
import os
from PIL import Image
import math

def yuv_import(filename,dims,numfrm,startfrm):  
    fp=open(filename,'rb')  
    blk_size = np.prod(dims) *3/2  
    fp.seek(int(blk_size)*startfrm,0)  
    Y=[]  
    U=[]  
    V=[]  
    #print (dims[0])
    #print (dims[1])
    d00=dims[0]//2  
    d01=dims[1]//2  
    #print (d00)
    #print (d01)  
    Yt=np.zeros((dims[0],dims[1]),'uint8','C')  
    Ut=np.zeros((d00,d01),'uint8','C')  
    Vt=np.zeros((d00,d01),'uint8','C')  
    for i in range(numfrm):  
        for m in range(dims[0]):  
            for n in range(dims[1]):  
                #print m,n  
                Yt[m,n]=ord(fp.read(1))  
        for m in range(d00):  
            for n in range(d01):  
                Ut[m,n]=ord(fp.read(1))  
        for m in range(d00):  
            for n in range(d01):  
                Vt[m,n]=ord(fp.read(1))  
        Y=Y+[Yt]  
        U=U+[Ut]  
        V=V+[Vt]  
    fp.close()  
    return (Y,U,V)  

if __name__ == '__main__':
    
    #call GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '4' 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    sess = tf.Session(config = config)
    
    quality=32
    
    img = Image.open(r"/home/zanz/AR-CNN-master/encoder/100098.png")#Maybe you need to change the directory
    w = img.size[0]
    pad_w = 4 - w % 4
    h = img.size[1]
    pad_h = 4 - h % 4
    
    os.system('ffmpeg -y -i %s -vf pad=%d:%d:0:0 %s'%('100098.png',w+pad_w,h+pad_h,'image_pad.png'))
    os.system('ffmpeg -y -i %s -s %dx%d -pix_fmt yuv420p %s'%('image_pad.png',w+pad_w,h+pad_h,'image.yuv'))
    os.system('ffmpeg -y -s %dx%d -i %s -vcodec libx264 -qp %d %s'%(w+pad_w,h+pad_h,'image.yuv',quality,'image_comp.264'))
    os.system('ffmpeg -y -i %s -vcodec rawvideo -s %dx%d %s'%('image_comp.264',w+pad_w,h+pad_h,'image_comp.yuv'))
    
    width=w+pad_w
    height=h+pad_h
    print ('width=%d,height=%d'%(width,height))
    img=yuv_import('image.yuv',(height,width),1,0) 
    #print(np.asarray(img[0][0]).shape)
    img_Y = img[0][0]
    img_Y = np.asarray(img_Y,np.float32)
    img_Y = img_Y.reshape([1, height, width,1]) /255 *100

    
    img_compressed=yuv_import('image_comp.yuv',(height,width),1,0) 
    img_compressed_Y = img_compressed[0][0]
    img_compressed_Y = np.asarray(img_compressed_Y,np.float32)
    img_compressed_Y = img_compressed_Y.reshape([1, height, width,1]) /255 *100

    reconstruct_Y = np.zeros([1,height, width,1])
    
    #restore ckpt
    saver=tf.train.import_meta_graph('/home/zanz/AR-CNN-master/encoder/model_x264/ckpt_quality=2/ckpt_03/model.ckpt.meta')#Maybe you need to change the directory
    saver.restore(sess,tf.train.latest_checkpoint("/home/zanz/AR-CNN-master/encoder/model_x264/ckpt_quality=2/ckpt_03/"))
    
    feed_data = {'MyInputCompres:0':img_compressed_Y,'MyInputTruth:0':img_Y}
    
    reconstruct_Y,loss,ori_loss = sess.run(['reconstruction/fan_out/MyOutput:0','loss/loss:0','original_loss/original_loss:0'], feed_dict=feed_data)
    print (reconstruct_Y.shape)
    
    PSNR_rec = 10.0 * math.log(65025.0 / loss) / math.log(10.0)
    PSNR_x264 = 10.0 * math.log(65025.0 / ori_loss) / math.log(10.0)
    print ('Loss: %f, Original_loss: %f, Loss decrease: %.3f%%' % (loss,ori_loss,((ori_loss - loss)/ ori_loss * 100)))
    print ('PSNR_rec: %fdB, PSNR_x264: %fdB, PSNR enhancement: %.3fdB' % (PSNR_rec,PSNR_x264,(PSNR_rec - PSNR_x264)))

    reconstruct_Y = reconstruct_Y * 255 /100
    
    reconstruct_Y = reconstruct_Y.reshape([height,width])
    for i in range(height):
        for j in range(width):
            if reconstruct_Y[i][j] > 255.0:
                reconstruct_Y[i][j] = 255.0
            if reconstruct_Y[i][j] < 0.0:
                reconstruct_Y[i][j] = 0.0
    print (reconstruct_Y.shape) 
