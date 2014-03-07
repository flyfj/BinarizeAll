function [ traincodes, trainlabels, testcodes, testlabels ] = loadfacecodes( codename, nbits, istrain )
%LOADFACECODES Summary of this function goes here
%   load precomputed youtube face dataset codes

datadir = 'C:\Users\jiefeng\Dropbox\hash_data\data/face_codes/';
%'data/face_codes/';%
traincodefile = sprintf('%s%s_train_%d.mat', datadir, codename, nbits);
load(traincodefile);
traincodes = Y;

testcodefile = sprintf('%s%s_test_%d.mat', datadir, codename, nbits);
load(testcodefile);
testcodes = tY;

labelfile = sprintf('%slabels.mat', datadir);
load(labelfile);
trainlabels = traingnd;
testlabels = testgnd;

end

