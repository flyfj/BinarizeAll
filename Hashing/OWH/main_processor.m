

function [base_pr, learn_pr] = main_processor(use_data, code_type)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main entrance for owh
% 1. load data
% 2. compute hash code using different methods
% 3. learn weights
% 4. evaluation

disp(['Dataset: ' num2str(use_data) ' Code: ' num2str(code_type)]);

%% init
addpath(genpath('svm'));

%% load raw data

disp('Loading raw feature data...');

% regular ml data format, treat as two groups, one for the same class, the
% other for all different ones
%use_data = 3;

[traindata, trainlabels] = loadTrainingData(use_data);
% make label starts from 1
trainlabels = trainlabels + 1;


% use subset as test data
testdata = [];
testlabels = [];

testids = [];

% separate into groups with same labels
unique_label_num = length( unique(trainlabels) );
train_groups = cell(unique_label_num, 1);
test_groups = cell(unique_label_num, 1);
for i=1:unique_label_num
    cls_ids = find(trainlabels == i);
    train_ids = cls_ids(1:int32(length(cls_ids)*0.8), 1);
    test_ids = cls_ids(int32(length(cls_ids)*0.8)+1:end, 1);
    testids = [testids; test_ids];
    % 8-2
    train_groups{i,1} = cls_ids;
    test_groups{i,1} = test_ids;
    testdata = [testdata; traindata(test_ids, :)];
    testlabels = [testlabels; trainlabels(test_ids, :)];
end


disp('Loaded raw feature data...');

%% compute base hash code

disp('Computing binary code for features...');

% general binary code params
code_params.nbits = 32;

% code name | code path
codetypes = cell(4,2);
codetypes{1,1} = 'SH'; codetypes{1,2} = '../SH/';
codetypes{2,1} = 'ITQ'; codetypes{2,2} = '../ITQ/';
codetypes{3,1} = 'LSH'; codetypes{3,2} = '';
codetypes{4,1} = 'KSH'; codetypes{4,2} = '../KSH';

%code_type = 3;
traincodes = [];
testcodes = [];

loadKSH = 1;

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
    % test codes
    [testcodes, U] = compressSH(testdata, sh_params);
    % convert to 0/1 array
    testcodes_str = [];
    for i=1:size(testcodes, 2)
        testcodes_str = [testcodes_str dec2bin(testcodes(:,i), 8)];
    end
    testcodes = testcodes_str - '0';
    
elseif code_type == 2
    % learn itq
    XX = traindata;
    sampleMean = mean(XX,1);
    XX = (XX - repmat(sampleMean,size(XX,1),1));
    % PCA
    [pc, l] = eigs(cov(XX(:,:)),code_params.nbits);
    XX = XX * pc;
    % ITQ
    [traincodes, R] = ITQ(XX(:,:),50);
    XX = XX*R;
    traincodes = zeros(size(XX));
    traincodes(XX>=0) = 1;
    traincodes = compactbit(traincodes>0);
    traincodes_str = [];
    for i=1:size(traincodes, 2)
        traincodes_str = [traincodes_str dec2bin(traincodes(:,i), 8)];
    end
    traincodes = traincodes_str - '0';
    
    % testcodes; take from traincodes
    testcodes = traincodes(testids, :);
    
elseif code_type == 3
   
    % lsh: random hash function from normal distribution
    lsh_params.nbits = code_params.nbits;
    lsh_params.funcs = randn(lsh_params.nbits, size(traindata, 2));
    traincodes = (lsh_params.funcs * traindata')';
    traincodes( traincodes>0 ) = 1;
    traincodes( traincodes<0 ) = 0;
    % test codes
    testcodes = (lsh_params.funcs * testdata')';
    testcodes( testcodes>0 ) = 1;
    testcodes( testcodes<0 ) = 0;
    
elseif code_type == 4
    
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


disp('Computing binary code for features done.');

%% generate similarity pairs

disp('Generating training pairs...');

sim_data = genSimData(train_groups, 'triplet');

disp('Generating training pairs done.');

%% learn weights using ranksvm formulation

disp('Learning weights...');

% now use relative attribute code

svm_type = 'ranksvm';

% construct parameters for svm code
svm_opt.lin_cg = 0; % not use conjugate gradient
svm_opt.iter_max_Newton = 200;   % Maximum number of Newton steps
svm_opt.prec = 0.0000000001;    %   prec: Stopping criterion
w_0 = ones(1, code_params.nbits);   % initial weights
W = [];

if strcmp(svm_type, 'ranksvm')
    
    % pre_compute hamming distance vector for selected pairs
    % to cope with svm code, each pair will be an invididual code sample

    % for triplet use
    triplet_num = length(sim_data{1,1});
    code_dist_vecs = zeros(3*triplet_num, code_params.nbits);
    ordered_idx = zeros(triplet_num, 2);
    sim_idx = zeros(triplet_num, 2);
    for i=1:triplet_num
        % compute similar pair distance
        code_dist_vecs(i*3-2,:) = abs( traincodes(sim_data{1,1}(i,2), :) - traincodes(sim_data{1,1}(i,4), :) );
        code_dist_vecs(i*3-1,:) = abs( traincodes(sim_data{1,1}(i,2), :) - traincodes(sim_data{1,1}(i,6), :) );
        code_dist_vecs(i*3, :) = abs( traincodes(sim_data{1,1}(i,2), :) - traincodes(sim_data{1,1}(i,8), :) );
        % add new sample ids to triplets: sim_code_dist, dis_code_dist
        sim_idx(i,:) = [i*3-2 i*3-1];
        ordered_idx(i,:) = [i*3-2 i*3];
    end
    
    % construct ordering and similarity matrix: pair_num X sample_num
    O = zeros(triplet_num, size(code_dist_vecs, 1));
    S = zeros(triplet_num, size(code_dist_vecs, 1));
    % Each row of O should contain exactly one +1 and one -1.
    for i=1:length(sim_idx)

        S(i, sim_idx(i,1)) = -1;
        S(i, sim_idx(i,2)) = 1;
    end

    for i=1:length(ordered_idx)

        O(i, ordered_idx(i,1)) = -1;
        O(i, ordered_idx(i,2)) = 1;
    end

    % use rank-svm first
    C_S = ones(1,length(sim_data{1,1})) * 0.1;
    C_O = ones(1,length(sim_data{1,1})) * 0.1;
    %W = ranksvm(code_dist_vecs, O, C_O', w_0', svm_opt); 

    % online mode
    
    W = w_0';
%     step = 1000;
%     for id=1:step:size(O,1)-step
%         tic
%         curO = O(id:id+step,:);
%         curS = S(id:id+step,:);
%         W = ranksvm_with_sim(code_dist_vecs, curO, curS, C_O(1,id:id+step)', C_S(1,id:id+step)', W, svm_opt);
%         toc
%         %disp(['Iter: ' num2str(id)]);
%     end
    
    W = ranksvm_with_sim(code_dist_vecs, O, S, C_O', C_S', w_0', svm_opt);
    %W = weightLearnerRank(w_0', code_dist_vecs, ordered_idx);
   
elseif strcmp(svm_type, 'normal')
    
    % use hamming distance vector as sample
    global X;
    X = zeros(length(sim_data{1,1})*4, code_params.nbits);
    newlabels = zeros(size(X,1), 1);
    for i=1:length(sim_data{1,1})
        X(i*4-3,:) = abs(traincodes(sim_data{1,1}(i,2), :) - traincodes(sim_data{1,1}(i,4), :));
        newlabels(i*4-3,1) = 1;
        X(i*4-2,:) = abs(traincodes(sim_data{1,1}(i,2), :) - traincodes(sim_data{1,1}(i,8), :));
        newlabels(i*4-2,1) = -1;
        X(i*4-1,:) = abs(traincodes(sim_data{1,1}(i,2), :) - traincodes(sim_data{1,1}(i,6), :));
        newlabels(i*4-1,1) = 1;
        X(i*4,:) = abs(traincodes(sim_data{1,1}(i,4), :) - traincodes(sim_data{1,1}(i,8), :));
        newlabels(i*4,1) = -1;
    end
    
    trainsz = int32(size(X,1)*0.8);
    svmmodel = svmtrain(double(newlabels(1:trainsz,:)), double(X(1:trainsz,:)));
    
    %svmoption = statset('Display', 'iter');
    %svmmodel = svmtrain(X, newlabels, 'kernel_function', 'quadratic', 'showplot', 0, 'options', svmoption);
    %[W,b0,obj] = primal_svm(1,newlabels,1,svm_opt)
    
    % test classification performance
    
    %pred_labels = svmclassify(svmmodel, X);
    [pred_labels, accuracy, scores] = svmpredict(newlabels(trainsz+1:end,:), X(trainsz+1:end,:), svmmodel);
    
    
    % test a query
    testid = 44;
    test_dist_vecs = repmat(traincodes(testid,:), size(traincodes, 1), 1);
    test_dist_vecs = abs(test_dist_vecs - traincodes);
    truelabels = -ones(size(traincodes,1), 1);
    truelabels(train_groups{trainlabels(testid,1), 1}) = 1;
    [pred_labels, accuracy, scores] = svmpredict(truelabels, test_dist_vecs, svmmodel);
%     corr_num = intersect( find(pred_labels==1), train_groups{trainlabel(testid,1), 1} );
%     corr_num = length(corr_num) / length(train_groups{trainlabel(testid,1), 1});
    
    % use scores to simply evaluate ranking performance
    [scores_sorted, score_sorted_idx] = sort(scores, 1, 'descend');
    score_inters = zeros(2, size(traincodes, 1));
    for i=1:size(traincodes, 1)
        % intersection value
        inter_num = length( intersect( score_sorted_idx(1:i, 1), train_groups{trainlabels(testid, 1), 1} ) );
        % precision
        score_inters(1,i) = double(inter_num) / i;
        % recall
        score_inters(2,i) = double(inter_num) / size(train_groups{trainlabels(testid, 1), 1}, 1);
    end

    % draw precision curve
    xlabel('Recall')
    ylabel('Precision')
    hold on
    axis([0 1 0 1]);
    hold on
    plot(score_inters(2,:), score_inters(1,:), 'r-')
    hold on
    legend('Clf')
    pause
    
end


disp('Weights learned.');

%% evaluation

showres = 0;
% use weights and no weights to compute ranking list for one sample first
% use base code: dist and cls_id
w1 = ones(code_params.nbits, 1);

validConstraintNum(traincodes, w1, sim_data)
validConstraintNum(traincodes, W, sim_data)

imgsz = 32;

% every two columns represent one test sample
numtest = 30;
%pickids = testlabels(1:numtest, :);
pickids = randsample(test_groups{1,1}, numtest);
step = 500;
base_pr = zeros(size(traincodes, 1)/step, 2*numtest);
learn_pr = zeros(size(traincodes, 1)/step, 2*numtest);

for i=1:numtest
    % process current code
    testlabel = trainlabels(pickids(i), :);%pickids(1,i)
    testsamp = traincodes(pickids(i),:);
    
    if(showres==1)
        % show test sample
        figure('Name', 'query')
        img = testdata(i, :)';
        img = reshape(img, imgsz, imgsz);
        imshow(img)
        hold on
        pause
    end
    
    base_dists = weightedHam(testsamp, traincodes, w1');
    [base_sorted_dist, base_sorted_idx] = sort(base_dists, 2);
    
    if(showres==1)
        % show ranked results in images, top 5
        figure('Name', 'Base Results')
        for k=1:5
            res = traindata(base_sorted_idx(1,k), :);
            res = reshape(res, imgsz, imgsz);
            subplot(1,5,k)
            imshow(res)
            hold on
        end
    end
    
    % use learned weights
    learn_dists = weightedHam(testsamp, traincodes, W');
    [learn_sorted_dist, learn_sorted_idx] = sort(learn_dists, 2);

    if(showres==1)
        figure('Name', 'Our Results')
        for k=1:5
            res = traindata(learn_sorted_idx(1,k), :);
            res = reshape(res, imgsz, imgsz);
            subplot(1,5,k)
            imshow(res)
            hold on
        end
        pause
        close all
    end
    
    % compute pr values
    cnt = 1;
    for j=1:step:size(traincodes, 1)
        % intersection value
        base_inter_num = size( intersect( base_sorted_idx(1, 1:j), train_groups{testlabel, 1} ), 1 );
        learn_inter_num = size( intersect( learn_sorted_idx(1, 1:j), train_groups{testlabel, 1} ), 1 );
        % precision
        base_pr(cnt,2*i-1) = double(base_inter_num) / j;
        learn_pr(cnt,2*i-1) = double(learn_inter_num) / j;
        % recall
        base_pr(cnt,2*i) = double(base_inter_num) / size(train_groups{testlabel, 1}, 1);
        learn_pr(cnt,2*i) = double(learn_inter_num) / size(train_groups{testlabel, 1}, 1);
        cnt = cnt + 1;
    end
    
end

% compute average pr
p_ids = 1:2:size(base_pr,2);
r_ids = 2:2:size(base_pr,2);
base_pr = [mean(base_pr(:,p_ids), 2), mean(base_pr(:,r_ids), 2)];
learn_pr = [mean(learn_pr(:,p_ids), 2), mean(learn_pr(:,r_ids), 2)];

end




