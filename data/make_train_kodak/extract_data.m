function extract_data()
    % train data
    process_train_data('train', 32, 10, 10);%setname, sub_img_size, stride, quality
    process_train_data('train', 32, 10, 20);
    process_train_data('train', 32, 10, 30);
    process_train_data('train', 32, 10, 40);
    
    process_train_data('test', 32, 10, 10);
    process_train_data('test', 32, 10, 20);
    process_train_data('test', 32, 10, 30);
    process_train_data('test', 32, 10, 40);
    
    process_train_data('val', 32, 10, 10);
    process_train_data('val', 32, 10, 20);
    process_train_data('val', 32, 10, 30);
    process_train_data('val', 32, 10, 40);
    

end

