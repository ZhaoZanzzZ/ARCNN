import os
import numpy as np
from PIL import Image
import struct

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

def makepaths():
    data_dir = '../ProcessedData_x264/train';
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    pad_path = '../ProcessedData_x264/pad'
    if not os.path.exists(pad_path):
        os.makedirs(pad_path)

    yuv_path = '../ProcessedData_x264/yuv'
    if not os.path.exists(yuv_path):
        os.makedirs(yuv_path)
        
    x264_path = '../ProcessedData_x264/264'
    if not os.path.exists(x264_path):
        os.makedirs(x264_path)
        
    compyuv_path = '../ProcessedData_x264/compyuv'
    if not os.path.exists(compyuv_path):
        os.makedirs(compyuv_path)
        
def writebin(data,height,width,bin_file):
    data=np.reshape(data,[height*width])
    for x in data:
        a = struct.pack('B',x)
        bin_file.write(a)

def process_train_data(setname, sub_img_size, stride, quality):
    data_dir = '../ProcessedData_x264/train';
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    bin_path = data_dir + '/'+ setname+ '_'+str(quality)+'.bin'
    bin_file = open(bin_path, 'wb')
    
    srcset_path = '../../'+'data/'+'BSDS500/'+ 'data/'+ 'images/'+setname#Maybe you need to change the directory
    fileNames = os.listdir(srcset_path)
    print (len(fileNames))
    
    for i in range(len(fileNames)):
        image_path = '/'+fileNames[i]
        image = Image.open(srcset_path + image_path)
        #image.show()
        w = image.size[0]
        #print (w)
        pad_w = 4 - w % 4
        h = image.size[1]
        pad_h = 4 - h % 4
        os.system('ffmpeg -y -i %s -vf pad=%d:%d:0:0:pink %s'%(srcset_path + image_path,w+pad_w,h+pad_h,'../ProcessedData_x264/pad/'+'pad'+fileNames[i]))
        os.system('ffmpeg -y -i %s -s %dx%d -pix_fmt yuv420p %s'%('../ProcessedData_x264/pad/'+'pad'+fileNames[i],w+pad_w,h+pad_h,'../ProcessedData_x264/yuv/'+'src'+fileNames[i]+'.yuv'))
        os.system('ffmpeg -y -s %dx%d -i %s -vcodec libx264 -preset fast -q:v %d %s'%(w+pad_w,h+pad_h,'../ProcessedData_x264/yuv/'+'src'+fileNames[i]+'.yuv',quality,'../ProcessedData_x264/264/'+'comp'+fileNames[i]+'.264'))
        os.system('ffmpeg -y -i %s -vcodec rawvideo -s %dx%d %s'%('../ProcessedData_x264/264/'+'comp'+fileNames[i]+'.264',w+pad_w,h+pad_h,'../ProcessedData_x264/compyuv/'+'comp'+fileNames[i]+'.yuv'))
        
        width=w+pad_w
        height=h+pad_h
        img=yuv_import('../ProcessedData_x264/yuv/'+'src'+fileNames[i]+'.yuv',(height,width),1,0) 
        img_Y = img[0][0]
        img_compressed=yuv_import('../ProcessedData_x264/compyuv/'+'comp'+fileNames[i]+'.yuv',(height,width),1,0) 
        img_compressed_Y = img_compressed[0][0]
        #print (np.array(img_Y).shape)
        
        row_num = ((height - sub_img_size) // stride) + 1
        col_num = ((width - sub_img_size) // stride) + 1
        row_shift = (height - ((row_num - 1) * stride + sub_img_size)) // 2
        col_shift = (width - ((col_num - 1) * stride + sub_img_size)) // 2
        
        for x in range (1,row_num+1):
            x_start = row_shift + (x - 1) * stride + 1
            x_end = row_shift + (x - 1) * stride + sub_img_size
            for y  in range (1,col_num+1):
                y_start = col_shift + (y - 1) * stride + 1
                y_end = col_shift + (y - 1) * stride + sub_img_size
                #print (img_Y.shape) 
                sub_img = img_Y[x_start : x_end+1, y_start:y_end+1]
                #print (sub_img.shape)
                sub_img_compressed = img_compressed_Y[x_start:x_end+1, y_start:y_end+1]
                
                writebin(sub_img,sub_img_size,sub_img_size,bin_file)
                writebin(sub_img_compressed,sub_img_size,sub_img_size,bin_file)
                
    bin_file.close()
    
makepaths()
process_train_data('train', 32, 10, 5)
process_train_data('test', 32, 10, 5)
process_train_data('val', 32, 10, 5)
print ('finished')