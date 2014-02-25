

%% compute binary codes for each dataset

use_data = 2;
dataname = 'cifar';

% load raw features
[traindata, trainlabels] = loadTrainingData(use_data);

[n, d] = size(traindata);


%% compute base hash code

disp('Computing binary code for features...');

% code name | code path
codetypes = cell(5,2);
codetypes{1,1} = 'sh'; codetypes{1,2} = '../SH/';
codetypes{2,1} = 'itq'; codetypes{2,2} = '../ITQ/';
codetypes{3,1} = 'lsh'; codetypes{3,2} = '../unsupervised_hash_code/';
codetypes{4,1} = 'mdsh'; codetypes{4,2} = '../unsupervised_hash_code/';
codetypes{5,1} = 'ksh'; codetypes{5,2} = '../KSH';

% extract all kinds of codes
codes = [1, 2, 3, 4];
bits = [16, 32, 48, 96, 128];

for id=1:length(codes)
    
    for j=1:length(bits)
        
        codeid = codes(id);
        code_params.nbits = bits(j);
        savefile = sprintf('data/%s_%s_%db.mat', dataname, codetypes{codeid,1}, bits(j));

        %code_type = 3;
        traincodes = [];
        testcodes = [];

        loadKSH = 1;

        % add code path
        addpath(genpath(codetypes{codeid, 2}));

        if codeid == 1
            % learn sh codes
            sh_params.nbits = code_params.nbits;
            sh_params = trainSH(traindata, sh_params);
            [traincodes, ~] = compressSH(traindata, sh_params);
            % convert to 0/1 array
            traincodes_str = [];
            for i=1:size(traincodes, 2)
                traincodes_str = [traincodes_str dec2bin(traincodes(:,i), 8)];
            end
            traincodes = traincodes_str - '0';
            traincodes = single(traincodes);
        %     % test codes
        %     [testcodes, ~] = compressSH(testdata, sh_params);
        %     % convert to 0/1 array
        %     testcodes_str = [];
        %     for i=1:size(testcodes, 2)
        %         testcodes_str = [testcodes_str dec2bin(testcodes(:,i), 8)];
        %     end
        %     testcodes = testcodes_str - '0';

        elseif codeid == 2

            % learn itq
            XX = traindata;
            sampleMean = mean(XX,1);
            XX = (XX - repmat(sampleMean,size(XX,1),1));
            % PCA
            [pc, l] = eigs(cov(XX(:,:)),code_params.nbits);
            XX = XX * pc;
            % ITQ
            [~, R] = ITQ(XX(:,:),50);
            XX = XX*R;
            traincodes = zeros(size(XX));
            traincodes(XX>=0) = 1;
            traincodes = compactbit(traincodes>0);
            traincodes_str = [];
            for i=1:size(traincodes, 2)
                traincodes_str = [traincodes_str dec2bin(traincodes(:,i), 8)];
            end
            traincodes = traincodes_str - '0';
            traincodes = single(traincodes);

            % testcodes; take from traincodes
        %     testcodes = traincodes(testids, :);

        elseif codeid == 3

            % lsh: random hash function from normal distribution
            lsh_params.nbits = code_params.nbits;
            lsh_params.funcs = randn(d, lsh_params.nbits);
            traincodes = traindata * lsh_params.funcs;
            % mean + bias
            meanv = mean(traincodes,1); 
            traincodes = traincodes-repmat(meanv, n, 1); % substract mean
            t = max(abs(traincodes),[],1);
            thres = rand(1,lsh_params.nbits).*t;   % generate threshold / bias
            traincodes = traincodes + repmat(thres, n, 1);

            traincodes = traincodes > 0;
            traincodes = single(traincodes);
            % test codes
        %     testcodes = (lsh_params.funcs * testdata')';
        %     testcodes = testcodes > 0;

        elseif codeid == 4

            % mdsh
            Sigma = 0.4;
            SHparamNew.nbits = code_params.nbits; % number of bits to code each sample
            SHparamNew.sigma = Sigma; % Sigma for the affinity. Different codes for different sigmas!
            SHparamNew = trainMDSH(traindata, SHparamNew);
            [B,traincodes] = compressMDSH(traindata, SHparamNew);
            traincodes = traincodes > 0;
            triancodes = single(traincodes);

        elseif codeid == 5

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

        save(savefile, 'traincodes', 'trainlabels');

        disp(['Saved code to ' savefile]);
        
    end
    
end

