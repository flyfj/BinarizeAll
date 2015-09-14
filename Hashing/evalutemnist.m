
% load data
db_data = zeros(800, 784);
db_label = zeros(800, 1);
q_data = zeros(200, 784);
q_label = zeros(200, 1);

mnist_dir = 'E:\Datasets\Search\MNIST\train\';


for i=0:9
    cur_dir = [mnist_dir num2str(i) '\'];
    imgfns = dir([cur_dir '*.jpg']);
    ids = randperm(length(imgfns), 100);
    
    for j=1:100
        if j<=80
            img = imread([cur_dir imgfns(j)]);
            db_data(i*80+j, :) = img(:);
            db_label(i*80+j, 1) = i;
        else
            img = imread([cur_dir imgfns(j)]);
            q_data(i*80+j-80, :) = img(:);
            q_label(i*80+j, 1) = i;
        end
    end
end

% compute code
