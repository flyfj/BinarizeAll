%% compute all types of hash code for raw features and output to files
% for use in c++ program

addpath(genpath('SH'));
addpath(genpath('ITQ'));
addpath(genpath('OWH'));

% load feature data from file
feat_file = 'F:\Datasets\Recognition\CIFAR-10\cifar-10-binary\train_1_data.txt';

[feats labels] = loadfeatfile(feat_file);

% compute sh
SHparams.nbits = 32;
SHparams = trainSH(feats, SHparams);
[sh_codes, U] = compressSH(feats, SHparams);
% convert sh code to unified code rep: array of int 0/1


% compute weights
