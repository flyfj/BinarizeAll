function [ traindata, trainlabels ] = loadTrainingData( dataset_id )
%LOADTRAININGDATA Summary of this function goes here
%   Detailed explanation goes here

% dataset_name dataset_folder
datasets = cell(4,2);
datasets{1,1} = 'dummy'; datasets{1,2} = '';
datasets{2,1} = 'CIFAR10';  %datasets{2,2} = 'H:\Datasets\Recognition\CIFAR10\';
datasets{2,2} = 'F:\Datasets\Recognition\CIFAR-10\cifar-10-binary\';
datasets{3,1} = 'minst'; 
datasets{3,2} = 'F:\Datasets\Recognition\MINST\Processed\';
%datasets{3,2} = 'D:\Box Files\My Box Files\Experiments\MINST\Processed\';


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
elseif dataset_id == 2
    % load gist from file
    traindata = load([datasets{2,2} 'train_1_gist.txt']);
    trainlabels = load([datasets{2,2} 'train_1_label.txt']);
elseif dataset_id == 3
    % load handwritten character data (processed from JHU)
    % each file contains 1000 images
    traindata = zeros(9000, 28*28);
    trainlabels = zeros(9000, 1);
    for i=0:9
        datafile = [datasets{3,2} 'data' num2str(i)];
        fid=fopen(datafile, 'r'); %-- open the file corresponding to digit 8 
        for j=1:1000
            [imgdata N] = fread(fid, [28 28], 'uchar'); % colume order, to display, show transpose
            traindata(i*1000+j, :) = reshape(imgdata, 1, 28*28);
            trainlabels(i*1000+j, 1) = i;
        end
        fclose(fid);
    end
    
end


end

