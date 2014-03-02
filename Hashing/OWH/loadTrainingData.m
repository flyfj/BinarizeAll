
function [ traindata, trainlabels, testdata, testlabels ] = loadTrainingData( dataset_id )
%LOADTRAININGDATA Summary of this function goes here
%   Detailed explanation goes here

% dataset_name dataset_folder
datasets = cell(4,2);
datasets{1,1} = 'dummy'; 
datasets{1,2} = '';
datasets{2,1} = 'cifar';
datasets{2,2} = 'C:/Users/jiefeng/Dropbox/hash_data/data/cifar_split.mat';  %'E:\Datasets\Recognition\CIFAR-10\cifar-10-binary\';
datasets{3,1} = 'mnist';
datasets{3,2} = 'C:\Users\jiefeng\Dropbox\hash_data\MINST\Processed\';
datasets{4,1} = 'youtube_face';
datasets{4,2} = 'E:\Datasets\Recognition\YoutubeFcae\kdd_face_split.mat';


if dataset_id == 1
    
    % use nearest neighbor to set positive samples
    trainlabels = ones(1000, 1) * 2;
    [traindata, traindistmat] = gen_dummy_data(300, 100);
    neighbor_num = 100;
    [sorted_dist, sorted_idx] = sort(traindist, 2);
    similar_bound = mean( sorted_dist(:, neighbor_num) );
    % !not clear how to do label here!
    similar_ids = sorted_idx(traindistmat < similar_bound);
    trainlabels(similar_ids, 1) = 1;

end

if dataset_id == 2
    % load gist from file
    load(datasets{2,2})
    
%     traindata = traindata;
%     testdata = testdata;
    trainlabels = traingnd;
    testlabels = testgnd;
%     traindata = load([datasets{2,2} 'train_1_gist.txt']);
%     trainlabels = load([datasets{2,2} 'train_1_label.txt']);
end

if dataset_id == 3
    % load handwritten character data (processed from JHU)
    % each file contains 1000 images
%     alldata = zeros(10000, 28*28);
%     alllabels = zeros(10000, 1);
    traindata = zeros(5000, 28*28);
    trainlabels = zeros(5000, 1);
    testdata = zeros(5000, 28*28);
    testlabels = zeros(5000, 1);
    traincnt = 1;
    testcnt = 1;
    for i=0:9
        datafile = [datasets{3,2} 'data' num2str(i)];
        fid=fopen(datafile, 'r'); %-- open the file corresponding to digit 8 
        for j=1:1000
            [imgdata, ~] = fread(fid, [28 28], 'uchar'); % colume order, to display, show transpose
            if j<=500
                traindata(traincnt, :) = reshape(imgdata', 1, 28*28);
                trainlabels(traincnt, 1) = i+1;
                traincnt = traincnt + 1;
            end
            if j>500
                testdata(testcnt, :) = reshape(imgdata', 1, 28*28);
                testlabels(testcnt, 1) = i+1;
                testcnt = testcnt + 1;
            end
%             alldata(i*1000+j, :) = reshape(imgdata, 1, 28*28);
%             alllabels(i*1000+j, 1) = i+1;
        end

        
        fclose(fid);
    end
end    

if dataset_id == 4
    
    % load face dataset
    
    
end


end

