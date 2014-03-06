
% implementation of cvpr 2013 whrank simple version
% for samples in the same class, compute unbinarized hash values and
% estimate a gaussian distribution for each bit
% weight based on value and std


% load unbinarized hash values
dataname = 'cifar';
datadir = 'C:\Users\jiefeng\Dropbox\hash_data\';
codenames = {'sh', 'itq', 'lsh', 'mdsh', 'iso'};
bits = [16, 32, 64, 96, 128];

codes = [1, 2, 3, 5];

for id=1:length(codes)
    
    codename = codenames{codes(id)};
    
    for j=1:length(bits)
        
        datafile = sprintf('%sdata/%s_codes/%s_%s_%db_un.mat', datadir, dataname, dataname, codename, bits(j));
        savefile = sprintf('%sdata/whrank/%s_%s_%db_whrank.mat', datadir, dataname, codename, bits(j));
        
        load(datafile);
        [n, d] = size(traincodes);
        labels = unique(trainlabels);
        traingroups = cell(length(labels), 1);
        % randomly sample similar pairs
        pairnum = 20000;
        diffvals = zeros(pairnum, bits(j));
        for i=1:pairnum
            clsid = randsample(labels, 1);
            [sampids, ~] = find(trainlabels==clsid);
            samps = randsample(sampids, 2);
            diffvals(i,:) = traincodes(samps(1), :) - traincodes(samps(2), :);
        end
        
        fmu = mean(diffvals, 1);
        fstd = std(diffvals, 1);
        
        save(savefile, 'fmu', 'fstd');
        
        disp(['params saved to ' savefile]);
    end
end

