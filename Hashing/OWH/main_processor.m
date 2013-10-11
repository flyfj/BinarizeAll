
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main entrance for owh
% 1. load data
% 2. compute hash code using different methods
% 3. learn weights
% 4. evaluation


%% load raw data
% regular ml data format, treat as two groups, one for the same class, the
% other for all different ones
traindata = [];
trainlabel = [];

% dataset_name dataset_folder
datasets = cell(4,2);
datasets{1,1} = 'dummy'; datasets{1,2} = '';
datasets{2,1} = 'CIFAR10';  datasets{2,2} = 'F:\Datasets\Recognition\CIFAR-10\cifar-10-binary\';

use_data = 1;
if use_data == 1
    % use nearest neighbor to set positive samples
    trainlabel = ones(size(traindata,1), 1) * 2;
    [traindata, traindistmat] = gen_dummy_data(300, 100);
    neighbor_num = 100;
    [sorted_dist, sorted_idx] = sort(traindist, 2);
    similar_bound = mean( sorted_dist(:, neighbor_num) );
    % !not clear how to do label here!
    similar_ids = sorted_idx(traindistmat < similar_bound);
    trainlabel(similar_ids, 1) = 1;
else if use_data == 2
        % load gist from file
        traindata = load([datasets{2,2} 'train_1_gist.txt']);
        trainlabel = load([datasets{2,2} 'train_1_label.txt']);
    end
end

% group samples from same class together
dataandlabel = [trainlabel traindata];
[sorted_data] = sortrows(dataandlabel, 1);
trainlabel = sorted_data(:, 1);
traindata = sorted_data(:, 2:end);

unique_labels = length( unique(trainlabel) );

% separate into groups with same labels
train_groups = cell(length(unique_labels), 1);
for i=1:length(unique_labels)
    train_groups{i,1} = find(trainlabel == i);
end


%% compute base hash code

% general binary code params
code_params.nbits = 32;

% code name | code path
codetypes = cell(4,2);
codetypes{1,1} = 'SH'; codetypes{1,2} = '../SH/';
codetypes{2,1} = 'ITQ'; codetypes{2,2} = '../ITQ/';
codetypes{3,1} = 'LSH'; codetypes{3,2} = '';

code_type = 1;
traincodes = [];

% add code path
addpath(genpath(codetypes{code_type, 2}));
if code_type == 1
    % learn sh codes
    sh_params.nbits = code_params.nbits;
    sh_params = trainSH(traindata, sh_params);
    [traincodes, U] = compressSH(traindata, sh_params);
else if codetype == 2
        % learn itq
    end
end


%% generate similarity pairs
% format: (samp_id, sim_id, dis_id)

% randomly select subset from same class as positive, the rest as negative
pair_num = 1000;



