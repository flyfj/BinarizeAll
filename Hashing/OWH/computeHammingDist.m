function [ dist, dist_vec ] = computeHammingDist( code1, code2, weights )
%COMPUTEHAMMINGDIST Summary of this function goes here
%   code is array of integer

% compute hamming distance vector (absolute value of bit-wise difference)
dist_vec = abs(code1 - code2);

if ~isempty(weights)  
    dist = weights .* dist_vec;
else
end
    dist = sum(dist_vec, 2);
end

