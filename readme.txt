#ARCNN
This introductory tutorial is modified on the basis of ARCNN. If you want to read the original paper, please download it at 'http://viplab.fudan.edu.cn/vip/projects/h264enc/wiki/Ml_tutorial'.
This tutorial uses two data sets, BSDS500 and kodak, but for the kodak data set, only its preprocessing code is provided. The code for how to call the processed data during training is similar to BSDS500. You can modify the catalog in the code according to BSDS500.

## jpeg
### Data Processing

*Download [BSDS500]&[kodak](http://viplab.fudan.edu.cn/vip/projects/h264enc/wiki/Ml_tutorial) datasets,and Put the two folders BSDS500 and kodak in './data' directory

*change matlab working directory to 'the/repository/path/data/make_train_BSDS500'
*run 'extract_data' on matlab console

*change matlab working directory to 'the/repository/path/data/make_train_kodak'
*run 'extract_data' on matlab console

### training 

*'cd src_jpeg'
*'python train.py'

### testing 
*'python encoder_JPEG.py'


## x264
### Data Processing
 
*'cd src/x264'
*'python make_train_264.py'

### training

*'python train.py'

### testing

*'python encoder_x264.py'

## x265
### Data Processing
 
*'cd src/x265'
*'python make_train_265.py'

### training

*'python train.py'

### testing

*'python encoder_x265.py'




