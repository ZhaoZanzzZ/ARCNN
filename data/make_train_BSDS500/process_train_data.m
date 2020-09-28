function process_train_data(setname, sub_img_size, stride, quality)%eg('train', 32, 10, 10)�����ò�����ԭ����������ݼ��Ķ�����
    data_dir = fullfile('..', 'ProcessedData_BSDS500', 'train');%fullfile���ɵ�ַ�ַ���
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end
    
    bin_path = fullfile(data_dir, [setname, '_', num2str(quality), '.bin']);%num2strʵ������1���ַ���0001�ı任�����ַ��������ʽ�����ַ���
    bin_file = fopen(bin_path, 'wb');%wbΪ�򿪷�ʽ

    file_list = dir(fullfile('..', 'BSDS500', 'data', 'images', setname, '*.jpg'));
    for i = 1:length(file_list)
        img = imread(fullfile('..', 'BSDS500', 'data', 'images', setname, file_list(i).name));%imread()�������ص��� Mat,һ������
        imwrite(img, 'temp.jpg', 'Quality', quality);%����ͼ�����
        img_compressed = imread('temp.jpg');
        delete('temp.jpg');
        
        img_size = size(img);
        img_row = img_size(1);%���ؾ��������
        img_col = img_size(2);%���ؾ��������
        
        % convert to luminance channelת��Ϊ����ͨ��
        for x = 1:img_row
            for y = 1:img_col
                img(x, y, 1) = 0.299 * img(x, y, 1) + 0.587 * img(x, y, 2) + 0.114 * img(x, y, 3);%RGB=>Y
                img_compressed(x, y, 1) = 0.299 * img_compressed(x, y, 1) + 0.587 * img_compressed(x, y, 2) + 0.114 * img_compressed(x, y, 3);
            end
        end
        
        row_num = floor((img_row - sub_img_size) / stride) + 1;%��img_row=1000,sub_img_size=30,stride=10������row_num=98
        col_num = floor((img_col - sub_img_size) / stride) + 1;%col_num=98
        row_shift = floor((img_row - ((row_num - 1) * stride + sub_img_size)) / 2);%row_shift=0
        col_shift = floor((img_col - ((col_num - 1) * stride + sub_img_size)) / 2);%col_shift=0
        
        for x = 1:row_num
            x_start = row_shift + (x - 1) * stride + 1;
            x_end = row_shift + (x - 1) * stride + sub_img_size;
            for y = 1:col_num
                y_start = col_shift + (y - 1) * stride + 1;
                y_end = col_shift + (y - 1) * stride + sub_img_size;
                %y�ֱ�Ϊ1:30,11:40,21:50...
                sub_img = img(x_start:x_end, y_start:y_end, 1);
                sub_img_compressed = img_compressed(x_start:x_end, y_start:y_end, 1);
                fwrite(bin_file, reshape(sub_img, [1, sub_img_size * sub_img_size]), 'uchar');
                %reshape�ǰ����е�˳�����ת���ģ�Ҳ���ǵ�һ�ж��꣬���ڶ��У����д��
                fwrite(bin_file, reshape(sub_img_compressed, [1, sub_img_size * sub_img_size]), 'uchar');
                %fwrite(fileID,A) ������ A ��Ԫ�ذ���˳���� 8 λ�޷�����������ʽд��һ���������ļ���
                %�ö������ļ����ļ���ʶ�� fileID ָʾ��ʹ�� fopen �ɴ��ļ�����ȡ fileID ֵ�� 
                %��ȡ�ļ�������� fclose(fileID) ���ر��ļ���
            end
        end 
    end
    
    fclose(bin_file);
end

