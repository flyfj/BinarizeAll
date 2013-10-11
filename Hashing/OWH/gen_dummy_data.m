function [data, distmat] = gen_dummy_data( num_samps, feat_dim )
%GEN_DUMMY_DATA Summary of this function goes here
%   generate random training data

data_range = 100;
aspectratio = 0.5;
data = rand([num_samps, feat_dim]) .* data_range;
data(:,2) = aspectratio * data(:,2);

distmat = distMat(data);

end

