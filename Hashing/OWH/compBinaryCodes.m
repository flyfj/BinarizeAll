

%% compute binary codes for each dataset

clear

datanames = {'dummay', 'cifar', 'mnist'};
use_data = 2;
dataname = datanames{use_data};

datadir = 'C:\Users\jiefeng\Dropbox\hash_data\';
% datadir = '';

% load raw features
[traindata, trainlabels, testdata, testlabels] = loadTrainingData(use_data);

[n, d] = size(traindata);
[ntest, d] = size(testdata);

%% compute base hash code

disp('Computing binary code for features...');

% code name | code path
codetypes = cell(6,2);
codetypes{1,1} = 'sh'; codetypes{1,2} = '../unsupervised_hash_code/';
codetypes{2,1} = 'itq'; codetypes{2,2} = '../unsupervised_hash_code/';
codetypes{3,1} = 'lsh'; codetypes{3,2} = '../unsupervised_hash_code/';
codetypes{4,1} = 'mdsh'; codetypes{4,2} = '../unsupervised_hash_code/';
codetypes{5,1} = 'iso'; codetypes{5,2} = '../unsupervised_hash_code/';
codetypes{6,1} = 'ksh'; codetypes{6,2} = '../KSH';

% extract all kinds of codes
codes = [4];
bits = [16 32 64 96 128];

binarize = 0;

for id=1:length(codes)
    
    for j=1:length(bits)
        
        codeid = codes(id);
        code_params.nbits = bits(j);
        if binarize == 1
            savefile = sprintf('%sdata/%s_codes/%s_%s_%db.mat', datadir, dataname, dataname, codetypes{codeid,1}, bits(j));
        else
            savefile = sprintf('%sdata/%s_codes/%s_%s_%db_un.mat', datadir, dataname, dataname, codetypes{codeid,1}, bits(j));
        end
        
        %code_type = 3;
        traincodes = [];
        testcodes = [];

%         loadKSH = 1;

        % add code path
        addpath(genpath(codetypes{codeid, 2}));

        if codeid == 1
            
            % learn sh codes
            sh_params.nbits = code_params.nbits;
            sh_params = trainSH(traindata, sh_params);
            [~, traincodes] = compressSH(traindata, sh_params);
            if binarize == 1
                traincodes = single(traincodes > 0);
            end
            
            % test codes
            [~, testcodes] = compressSH(testdata, sh_params);
            if binarize
                testcodes = single(testcodes > 0);
            end
            
            clear sh_params
            
        end
        
        if codeid == 2

            % learn itq
            meanv = mean(traindata,1);
            X = traindata - repmat(meanv, n, 1);
            cov = X'*X;
            [U,V] = eig(cov);
            eigenvalue = diag(V)';
            [eigenvalue, order] = sort(eigenvalue, 'descend');
            W = U(:, order(1:code_params.nbits));
            traincodes = X*W;
            [~, R] = ITQ(traincodes, 50);
            W = W*R;
            meanv = meanv*W;
            traincodes = traincodes*R;
            if binarize == 1
                traincodes = single(traincodes>0);
            end
            
            testcodes = testdata*W - repmat(meanv, ntest, 1);
            if binarize == 1
                testcodes = single(testcodes>0);
            end
            clear meanv W R
            
            % test
%             w1 = ones(code_params.nbits, 1);
%             base_dists = weightedHam(testcodes, testcodes, w1', 0);
%             [~, base_sorted_idx] = sort(base_dists, 2);
%             nt = size(testcodes, 1);
%             ranked_labels = testlabels(base_sorted_idx);
%             interval = 20;
%             pt_num = 1 + floor(nt/interval);
%             prr = zeros(1, pt_num*2);
%             for pi = 1:nt
%                 h = double(ranked_labels(pi, :) == testlabels(pi));
%                 ind = find(h > 0);
%                 pn = length(ind); 
% 
%                 %% PR curve
%                 prr = prr + PR_new(h', interval);
%                 clear h;
%             end
% 
%             prr = prr' ./ nt;
%             plot(prr(pt_num+1:end), prr(1:pt_num), 'r-')
%             grid on
%             pause
            
        end
        
        if codeid == 3

            % lsh: random hash function from normal distribution
            lsh_params.nbits = code_params.nbits;
            lsh_params.funcs = randn(d, lsh_params.nbits);
%             traindata2 = bsxfun(@minus, traindata, mean(traindata));
            traincodes = traindata * lsh_params.funcs;
            % mean + bias
            meanv = mean(traincodes,1); 
            traincodes = traincodes-repmat(meanv, n, 1); % substract mean
            % not use threshold, keep code balanced
%             t = max(abs(traincodes),[],1);
%             thres = rand(1,lsh_params.nbits).*t;   % generate threshold / bias
%             traincodes = traincodes + repmat(thres, n, 1);

            if binarize == 1
                traincodes = single(traincodes > 0);
            end
            
            % test codes
%             testdata2 = bsxfun(@minus, testdata, mean(testdata));
            testcodes = testdata * lsh_params.funcs;
            % mean + bias
            meanv = mean(testcodes,1); 
            testcodes = testcodes-repmat(meanv, ntest, 1); % substract mean
%             t = max(abs(testcodes),[],1);
%             thres = rand(1,lsh_params.nbits).*t;   % generate threshold / bias
%             testcodes = testcodes + repmat(thres, ntest, 1);
            if binarize == 1
                testcodes = single(testcodes > 0);
            end
            
            clear lsh_params
            
        end
        
        if codeid == 4

            % mdsh
            Sigma = 0.4;
            SHparamNew.nbits = code_params.nbits; % number of bits to code each sample
            SHparamNew.sigma = Sigma; % Sigma for the affinity. Different codes for different sigmas!
            SHparamNew = trainMDSH(traindata, SHparamNew);
            [~, traincodes] = compressMDSH(traindata, SHparamNew);
            if binarize == 1
                traincodes = sign(traincodes);
%                 traincodes = single(traincodes > 0);
            end

            [~, testcodes] = compressMDSH(testdata, SHparamNew);
            if binarize == 1
                testcodes = sign(testcodes);
%                 testcodes = single(testcodes > 0);
            end

        end
        
        if codeid == 5
            
            % iso hashing
            meanv = mean(traindata, 1);
            traindata = traindata - repmat(meanv,n,1);
            cov = traindata' * traindata;
            [U,V] = eig(cov);
            clear cov
            eigenvalue = diag(V)';
            [eigenvalue,order] = sort(eigenvalue, 'descend');
            W = U(:,order(1:code_params.nbits));
            eigenvalue = eigenvalue(1:code_params.nbits);

            R = GradientFlow(diag(eigenvalue));
            W = W*R;
            meanv = meanv*W;
            traincodes = traindata*W;
            if binarize == 1
                traincodes = single(traincodes > 0);
            end
            
            testcodes = testdata*W-repmat(meanv, ntest, 1);
            if binarize == 1
                testcodes = single(testcodes > 0);
            end
            
            clear W meanv
            
        end
        
        if codeid == 6

            % ksh
            if(loadKSH == 1)
                if(use_data == 2)
                    load cifar_ksh_32_300_1000;
                elseif(use_data == 3)
                    load mnist_ksh_32_300_1000;
                end
            else

                global m;
                m = 300;    % number of anchors
                global r;
                r = code_params.nbits;     % number of hash bits
                % sample anchors
                anchor_idx = randsample(1:size(traindata,1), m);
                anchor = traindata(anchor_idx, :);
                KTrain = sqdist(traindata',anchor');    % compute distance between each sample and anchor
                global sigma;
                sigma = mean(mean(KTrain,2));   % sigma
                KTrain = exp(-KTrain/(2*sigma));    % normalize?
                global mvec;
                mvec = mean(KTrain);    % mean
                KTrain = KTrain-repmat(mvec, size(traindata,1), 1);   % kernel value computation

                % pairwise label matrix
                % create a diagonal matrix with 1 and others with -1?
                % select subset as training sample
                global trn; % number of labeled training samples
                trn = 1000;
                label_index = randsample(1:size(traindata,1), trn);
                trngnd = trainlabels(label_index');    % 
                temp = repmat(trngnd,1,trn)-repmat(trngnd',trn,1);
                S0 = -ones(trn,trn);
                tep = temp == 0;
                S0(tep) = 1;
                clear temp;
                clear tep;
                S = r*S0;

                % projection optimization
                KK = KTrain(label_index',:);
                RM = KK'*KK; 
                A1 = zeros(m,r);
                flag = zeros(1,r);
                for rr = 1:r
                    [rr]
                    if rr > 1
                        S = S-y*y';
                    end

                    LM = KK'*S*KK;
                    [U,V] = eig(LM,RM);
                    eigenvalue = diag(V)';
                    [eigenvalue,order] = sort(eigenvalue,'descend');
                    A1(:,rr) = U(:,order(1));
                    tep = A1(:,rr)'*RM*A1(:,rr);
                    A1(:,rr) = sqrt(trn/tep)*A1(:,rr);
                    clear U;    
                    clear V;
                    clear eigenvalue; 
                    clear order; 
                    clear tep;  

                    [get_vec, cost] = OptProjectionFast(KK, S, A1(:,rr), 500);
                    y = double(KK*A1(:,rr)>0);
                    ind = find(y <= 0);
                    y(ind) = -1;
                    clear ind;
                    y1 = double(KK*get_vec>0);
                    ind = find(y1 <= 0);
                    y1(ind) = -1;
                    clear ind;
                    if y1'*S*y1 > y'*S*y
                        flag(rr) = 1;
                        A1(:,rr) = get_vec;
                        y = y1;
                    end
                end

                % encoding
                traincodes = single(A1'*KTrain' > 0)';

                % process testdata
                KTest = sqdist(testdata',anchor');
                KTest = exp(-KTest/(2*sigma));
                KTest = KTest-repmat(mvec, size(testdata,1), 1);
                testcodes = single(A1'*KTest' > 0)';

                % save
                if(use_data == 2)
                    save cifar_ksh_48_300_1000 traincodes testcodes A1 anchor mvec sigma
                elseif(use_data == 3)
                    save mnist_ksh_48_300_1000 traincodes testcodes A1 anchor mvec sigma
                end

            end

        end

        if codeid ~= 4
            save(savefile, 'traincodes', 'trainlabels', 'testcodes', 'testlabels');
        else
            save(savefile, 'traincodes', 'trainlabels', 'testcodes', 'testlabels', 'SHparamNew');
        end

        disp(['Saved code to ' savefile]);
        
    end
    
end

