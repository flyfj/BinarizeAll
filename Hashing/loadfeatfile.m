function [feats, labels] = loadfeatfile( file )
%LOADFEATFILE Summary of this function goes here
%   load feature vectors for each sample from file (ml data type)

% sample_num, feat_dim, cls_num
% label feat_vec

fid = fopen(file);
params = fscanf(fid, '%f %f %f', 3);
feats = fscanf(fid, '%f', [params(2)+1 params(1)]);
labels = feats(1,:)';
feats = feats(2:end, :)';
fclose(fid);

end

