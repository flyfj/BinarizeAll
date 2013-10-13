
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main entrance for owh
% 1. load data
% 2. compute hash code using different methods
% 3. learn weights
% 4. evaluation

%% init
addpath('svm');

%% load raw data
% regular ml data format, treat as two groups, one for the same class, the
% other for all different ones
traindata = [];
trainlabel = [];

% dataset_name dataset_folder
datasets = cell(4,2);
datasets{1,1} = 'dummy'; datasets{1,2} = '';
datasets{2,1} = 'CIFAR10';  datasets{2,2} = 'H:\Datasets\Recognition\CIFAR10\';

use_data = 2;
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

% make label starts from 1
trainlabel = trainlabel + 1;

% separate into groups with same labels
unique_label_num = length( unique(trainlabel) );
train_groups = cell(unique_label_num, 1);
for i=1:unique_label_num
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
    % convert to 0/1 array
    traincodes_str = [];
    for i=1:size(traincodes, 2)
        traincodes_str = [traincodes_str dec2bin(traincodes(:,i), 8)];
    end
    traincodes = traincodes_str - '0';
else if codetype == 2
        % learn itq
    end
end


%% generate similarity pairs
% format: (samp_id, sim_id, dis_id)

% randomly select subset from same class as positive, the rest as negative
triplet_num = 1000;
sim_triplets = zeros(triplet_num, 6);
for i=1:triplet_num
    % select a sample
    samp_cls_id = int32( randsample(unique_label_num, 1) );
    samp_obj_id = int32( randsample(train_groups{samp_cls_id}, 1) );
    % select similar sample from same class
    sim_cls_id = samp_cls_id;
    sim_obj_id = 0;
    while 1
        temp_obj_id = int32( randsample(train_groups{samp_cls_id}, 1) );
        if temp_obj_id ~= samp_obj_id
            sim_obj_id = temp_obj_id;
            break;
        end
    end
    % select dissimilar sample from different classes
    dis_cls_id = 0;
    while 1
        temp_cls_id = int32( randsample(unique_label_num, 1) );
        if temp_cls_id ~= samp_cls_id
            dis_cls_id = temp_cls_id;
            break;
        end
    end
    dis_obj_id = int32( randsample(train_groups{dis_cls_id}, 1) );
    % add to collection
    sim_triplets(i,:) = [samp_cls_id, samp_obj_id, sim_cls_id, sim_obj_id, dis_cls_id, dis_obj_id];
end

% pre_compute hamming distance vector for selected pairs
% to cope with svm code, each pair will be an invididual code sample
code_dist_vecs = zeros(2*triplet_num, code_params.nbits);
added_idx = zeros(triplet_num, 2);
for i=1:triplet_num
    code_dist_vecs(i*2-1,:) = abs( traincodes(sim_triplets(i,2), :) - traincodes(sim_triplets(i,4), :) );
    code_dist_vecs(i*2,:) = abs( traincodes(sim_triplets(i,2), :) - traincodes(sim_triplets(i,6), :) );
    % add new sample ids to triplets: sim_code_dist, dis_code_dist
   added_idx(i,:) = [i*2-1 i*2];
end
sim_triplets = [sim_triplets added_idx];

%% learn weights using ranksvm formulation
% now use relative attribute code

% construct parameters for svm code
svm_opt.lin_cg = 0; % not use conjugate gradient
svm_opt.iter_max_Newton = 20;   % Maximum number of Newton steps
svm_opt.prec = 0.000001;    %   prec: Stopping criterion
w_0 = zeros(1,code_params.nbits);   % initial weights

% construct ordering and similarity matrix: pair_num X sample_num
O = zeros(triplet_num, size(code_dist_vecs, 1));
S = zeros(triplet_num, size(code_dist_vecs, 1));
% Each row of O should contain exactly one +1 and one -1.
for i=1:triplet_num
    
    O(i, sim_triplets(i,7)) = -1;
    O(i, sim_triplets(i,8)) = 1;
end

% use rank-svm first
C = ones(1,triplet_num) * 0.1;
W = ranksvm(code_dist_vecs, O, C', w_0', svm_opt); 

%w = ranksvm_with_sim(traincodes,O,S,C_O,C_S,w,opt);


