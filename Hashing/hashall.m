%% compute all types of hash code for raw features and output to files
% for use in c++ program

addpath(genpath('SH'));
addpath(genpath('ITQ'));
addpath(genpath('OWH'));

% load feature data from file
feat_file = 'F:\Datasets\Recognition\CIFAR-10\cifar-10-binary\test_data.txt';

[feats labels] = loadfeatfile(feat_file);

% params
nbits = 32;

% compute sh
SHparams.nbits = nbits;
SHparams = trainSH(feats, SHparams);
[sh_codes, U] = compressSH(feats, SHparams);
% convert sh code to unified code rep: array of int 0/1
% concatenate all char
sh_code_str = [];
for r=1:size(sh_codes,1)
    cur_row = [];
    for c=1:size(sh_codes,2)
        cur_code = dec2bin(sh_codes(r,c), 8);
        cur_row = [cur_row cur_code];
    end
    sh_code_str = [sh_code_str; cur_row];
end
sh_code_num = int8(sh_code_str-'0');

% hash code data file format:
% #samp #feat_dim #cls_num
% label code_Str
sh_data = double([labels sh_code_num]);
save('test.code', 'sh_data', '-ascii');



% learn weights
