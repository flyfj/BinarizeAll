
datafolder = 'H:\Datasets\Recognition\CIFAR10\cifar-10-batches-mat\';
datafile = [datafolder 'test_batch.mat'];
savefolder = 'test\';

rawdata = load(datafile);
data = rawdata.data;
labels = double(rawdata.labels);

% extract image and save to folder
mkdir(datafolder, savefolder);
savefolder = [datafolder savefolder];

% for i=1:size(rawdata.data, 1)
%     
%     img = zeros(32, 32, 3);
%     img(:,:,1) = reshape(rawdata.data(i, 1:1024), 32, 32)';
%     img(:,:,2) = reshape(rawdata.data(i, 1025:2048), 32, 32)';
%     img(:,:,3) = reshape(rawdata.data(i, 2049:3072), 32, 32)';
%     img = uint8(img);
%     
%     imshow(img);
%     imwrite(img, [savefolder num2str(i) '.jpg'], 'jpeg');
%     
%     disp(['Finish' num2str(i)]);
% end

% save label file
save([savefolder 'label.txt'], 'labels', '-ascii');

