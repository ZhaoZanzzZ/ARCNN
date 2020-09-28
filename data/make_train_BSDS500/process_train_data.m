function process_train_data(setname, sub_img_size, stride, quality)%eg('train', 32, 10, 10)，设置步长的原因是提高数据集的多样性
    data_dir = fullfile('..', 'ProcessedData_BSDS500', 'train');%fullfile构成地址字符串
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end
    
    bin_path = fullfile(data_dir, [setname, '_', num2str(quality), '.bin']);%num2str实现数字1到字符串0001的变换，以字符数组的形式串联字符串
    bin_file = fopen(bin_path, 'wb');%wb为打开方式

    file_list = dir(fullfile('..', 'BSDS500', 'data', 'images', setname, '*.jpg'));
    for i = 1:length(file_list)
        img = imread(fullfile('..', 'BSDS500', 'data', 'images', setname, file_list(i).name));%imread()函数返回的是 Mat,一个矩阵
        imwrite(img, 'temp.jpg', 'Quality', quality);%保存图像矩阵
        img_compressed = imread('temp.jpg');
        delete('temp.jpg');
        
        img_size = size(img);
        img_row = img_size(1);%返回矩阵的行数
        img_col = img_size(2);%返回矩阵的列数
        
        % convert to luminance channel转换为亮度通道
        for x = 1:img_row
            for y = 1:img_col
                img(x, y, 1) = 0.299 * img(x, y, 1) + 0.587 * img(x, y, 2) + 0.114 * img(x, y, 3);%RGB=>Y
                img_compressed(x, y, 1) = 0.299 * img_compressed(x, y, 1) + 0.587 * img_compressed(x, y, 2) + 0.114 * img_compressed(x, y, 3);
            end
        end
        
        row_num = floor((img_row - sub_img_size) / stride) + 1;%设img_row=1000,sub_img_size=30,stride=10。则有row_num=98
        col_num = floor((img_col - sub_img_size) / stride) + 1;%col_num=98
        row_shift = floor((img_row - ((row_num - 1) * stride + sub_img_size)) / 2);%row_shift=0
        col_shift = floor((img_col - ((col_num - 1) * stride + sub_img_size)) / 2);%col_shift=0
        
        for x = 1:row_num
            x_start = row_shift + (x - 1) * stride + 1;
            x_end = row_shift + (x - 1) * stride + sub_img_size;
            for y = 1:col_num
                y_start = col_shift + (y - 1) * stride + 1;
                y_end = col_shift + (y - 1) * stride + sub_img_size;
                %y分别为1:30,11:40,21:50...
                sub_img = img(x_start:x_end, y_start:y_end, 1);
                sub_img_compressed = img_compressed(x_start:x_end, y_start:y_end, 1);
                fwrite(bin_file, reshape(sub_img, [1, sub_img_size * sub_img_size]), 'uchar');
                %reshape是按照列的顺序进行转换的，也就是第一列读完，读第二列，按列存放
                fwrite(bin_file, reshape(sub_img_compressed, [1, sub_img_size * sub_img_size]), 'uchar');
                %fwrite(fileID,A) 将数组 A 的元素按列顺序以 8 位无符号整数的形式写入一个二进制文件。
                %该二进制文件由文件标识符 fileID 指示。使用 fopen 可打开文件并获取 fileID 值。 
                %读取文件后，请调用 fclose(fileID) 来关闭文件。
            end
        end 
    end
    
    fclose(bin_file);
end

