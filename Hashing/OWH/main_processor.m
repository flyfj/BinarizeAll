
function [base_pr, learn_pr] = main_processor(dataname, codename, nbits, method)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main entrance for owh
% 1. load data
% 2. compute hash code using different methods
% 3. learn weights
% 4. evaluation

% method 0: base code; 1: learn weighted; 2: whrank; 3: weighted online


%% load binary codes

addpath(genpath('svm'));
addpath(genpath('../unsupervised_hash_code/'));

% dataname = 'mnist';
% codename = 'lsh';
% nbits = 32;
% method = 1;

datadir = 'C:\Users\jiefeng\Dropbox\hash_data\';
% datadir = '';

disp(['Dataset: ' dataname '; Code: ' codename '; bits: ' num2str(nbits)]);

disp('Loading binary codes...');

code_params.nbits = nbits;
codefile = sprintf('%sdata/%s_codes/%s_%s_%db.mat', datadir, dataname, dataname, codename, nbits);
uncodefile = sprintf('%sdata/%s_codes/%s_%s_%db_un.mat', datadir, dataname, dataname, codename, nbits);
basecurvefile = sprintf('%sres/%s_%s_%db_pr.mat', datadir, dataname, codename, nbits);
learncurvefile = sprintf('%sres/%s_%s_%db_pr_weighted.mat', datadir, dataname, codename, nbits);
whrankcurvefile = sprintf('%sres/%s_%s_%db_pr_whrank.mat', datadir, dataname, codename, nbits);

if method == 2
    % load whrank parameters and data
    whrankfile = sprintf('%sdata/whrank/%s_%s_%db_whrank.mat', datadir, dataname, codename, nbits);
    load(whrankfile);
    
    load(uncodefile);
    traincodes_un = traincodes;
    testcodes_un = testcodes;
end
    
if strcmp(dataname, 'face') == 1
    [traincodes, trainlabels, testcodes, testlabels] = loadfacecodes(codename, nbits, 1);
else
    load(codefile);
end

% load('cifar_split.mat')
% load('itq_48.mat');
% tep = find(Y<=0);
% Y(tep) = -1;
% 
% tn = size(testdata, 1);
% tY = testdata*W-repmat(mvec,tn,1);
% tY = (tY>0);
% tY = single(tY);
% tep = find(tY<=0);
% tY(tep) = -1;

labels = unique(trainlabels);

%split data into train and test: 50-50
traingroups = cell(length(labels), 1);
testgroups = cell(length(labels), 1);
big500 = 0;
big1000 = 0;
big1500 = 0;
big2000 = 0;
newtestcodes = [];
newtestlabels = [];
for i=1:length(labels)
    clsids = find(trainlabels == labels(i));
    traingroups{i,1} = clsids;
    clsids = find(testlabels == labels(i));
    testgroups{i,1} = clsids;%clsids(1:int32(length(clsids)/3));
    newtestlabels = [newtestlabels; testlabels(clsids(1:int32(length(clsids)/3)))];
    newtestcodes = [newtestcodes; testcodes(testgroups{i},:)];
    
    if(length(clsids) > 500)
        big500 = big500 + 1;
    end
    if(length(clsids) > 1000)
        big1000 = big1000 + 1;
    end
    if(length(clsids) > 1300)
        big1500 = big1500 + 1;
    end
          
end

testlimit = 550;   % maximum size of testgroup to look at
biggroup = 0;
validcls = [];
% form new groups
if strcmp(dataname, 'face') == 1
    for i=1:length(labels)
        clsids = find(newtestlabels == labels(i));
        if(length(clsids) > testlimit)
            biggroup = biggroup + 1;
            validcls = [validcls; i];
        end
        testgroups{i,1} = clsids;
    end
    
    testcodes = newtestcodes;
    testlabels = newtestlabels;
    clear newtestcodes newtestlabels
end


disp('Binary code loaded.');



%% generate similarity pairs
if method == 1
    disp('Generating training pairs...');

    if ~exist('sim_data', 'var')
        sim_data = genSimData(traingroups, 'triplet', 5000, validcls);
%         sim_data2 = genSimData(testgroups, 'triplet', 2000);
%         sim_data = [sim_data; sim_data2];
    end

    disp('Generating training pairs done.');
end

%% learn weights using ranksvm formulation

if method == 1

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
        triplet_num = size(sim_data, 1);
        code_dist_vecs = zeros(3*triplet_num, code_params.nbits);
        ordered_idx = zeros(triplet_num, 2);
        sim_idx = zeros(triplet_num, 2);
        cnt = 1;
        for i=1:triplet_num
            % compute similar pair distance
            code_dist_vecs(cnt,:) = abs( traincodes(sim_data(i,2), :) - traincodes(sim_data(i,4), :) );
            cnt = cnt + 1;
            code_dist_vecs(cnt,:) = abs( traincodes(sim_data(i,2), :) - traincodes(sim_data(i,6), :) );
            sim_idx(i,:) = [cnt cnt-1];
            cnt = cnt + 1;
            code_dist_vecs(cnt, :) = abs( traincodes(sim_data(i,2), :) - traincodes(sim_data(i,8), :) );
            ordered_idx(i,:) = [cnt cnt-2];
            cnt = cnt + 1;

        end

        % convert to -1 / 1
%         code_dist_vecs = double(2*code_dist_vecs - 1);
%         imagesc(code_dist_vecs)
%         pause

        % construct ordering and similarity matrix: pair_num X sample_num
        O = zeros(triplet_num, size(code_dist_vecs, 1));
        S = zeros(triplet_num, size(code_dist_vecs, 1));
        % Each row of O should contain exactly one +1 and one -1.
        for i=1:length(sim_idx)

            S(i, sim_idx(i,1)) = 1;
            S(i, sim_idx(i,2)) = -1;
        end

        for i=1:length(ordered_idx)

            O(i, ordered_idx(i,1)) = 1;
            O(i, ordered_idx(i,2)) = -1;
        end

        % use rank-svm first
        C_S = ones(1, triplet_num) * 100;
        C_O = ones(1, triplet_num) * 100;
%         W = ranksvm(code_dist_vecs, O, C_O', w_0', svm_opt); 

        % online mode
%         W = w_0';
%         step = 1000;
%         for id=1:step:size(O,1)-step
%             tic
%             curO = O(id:id+step,:);
%             curS = S(id:id+step,:);
%             W = ranksvm_with_sim(code_dist_vecs, curO, curS, C_O(1,id:id+step)', C_S(1,id:id+step)', W, svm_opt);
%             toc
%             %disp(['Iter: ' num2str(id)]);
%         end

        W = ranksvm_with_sim(code_dist_vecs, O, S, C_O', C_S', w_0', svm_opt);
        %W = weightLearnerRank(w_0', code_dist_vecs, ordered_idx);

    end

    if strcmp(svm_type, 'normal')

        % use hamming distance vector as sample
        global X;
        sampnum = size(sim_data{1,1}, 1);
        X = zeros(2*sampnum, code_params.nbits);
        newlabels = zeros(size(X,1), 1);

        cnt = 1;
        for i=1:sampnum
            % positive samples
            X(cnt,:) = abs(traincodes(sim_data{1}(i,2), :) - traincodes(sim_data{1}(i,4), :));
            newlabels(cnt,1) = 1;
            cnt = cnt + 1;
            % negative sample
            X(cnt,:) = abs(traincodes(sim_data{2}(i,2), :) - traincodes(sim_data{2}(i,4), :));
            newlabels(cnt,1) = -1;
            cnt = cnt + 1;
        end

        X = X*2 - 1;

        trainsz = int32(size(X,1)*0.9);
        svmmodel = svmtrain( double(newlabels(1:trainsz,:)), double(X(1:trainsz,:)), '-t 0');

        %svmoption = statset('Display', 'iter');
        %svmmodel = svmtrain(X, newlabels, 'kernel_function', 'quadratic', 'showplot', 0, 'options', svmoption);
        %[W,b0,obj] = primal_svm(1,newlabels,1,svm_opt)

        % test classification performance

        %pred_labels = svmclassify(svmmodel, X);
        [pred_labels, accuracy, scores] = svmpredict(newlabels(trainsz+1:end,:), X(trainsz+1:end,:), svmmodel);
        [poslabels, ~] = find( newlabels(trainsz+1:end, :) == 1 );
        length( find( pred_labels(poslabels, 1) == 1 ) ) / size(poslabels, 1)

        % test a query
        testid = 300;
        gtlabels = testgroups{testlabels(testid,1), 1};
        % compute difference vector with each testcode
        test_dist_vecs = repmat(testcodes(testid,:), size(testcodes, 1), 1);
        test_dist_vecs = abs(test_dist_vecs - testcodes) * 2 - 1;
        test_dist_vecs = double(test_dist_vecs);
        truelabels = -ones(length(testlabels), 1);
        truelabels(gtlabels) = 1;
        [pred_labels, accuracy, scores] = svmpredict(truelabels, test_dist_vecs, svmmodel);
        [poslabels, ~] = find( truelabels == 1 );
        length( find( pred_labels(poslabels, 1) == 1 ) ) / size(poslabels, 1)

        % use scores to simply evaluate ranking performance
        [scores_sorted, score_sorted_idx] = sort(scores, 1, 'descend');
        score_inters = zeros(2, size(testcodes, 1));
        for i=1:size(testcodes, 1)
            % intersection value
            inter_num = length( find( (testlabels(score_sorted_idx(1:i)) == testlabels(testid)) > 0) ); 
            % precision
            score_inters(1,i) = double(inter_num) / i;
            % recall
            score_inters(2,i) = double(inter_num) / length(gtlabels);
        end

        % draw precision curve
        close all
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

end

%% evaluation

clear traincodes

showres = 0;
% use weights and no weights to compute ranking list for one sample first
% use base code: dist and cls_id
w1 = ones(code_params.nbits, 1);

% validConstraintNum(traincodes, w1, sim_data)
% validConstraintNum(traincodes, W, sim_data)

imgsz = 32;

% every two columns represent one test sample

%pickids = testlabels(1:numtest, :);
% pickids = randsample(testgroups{1,1}, numtest);

% ptnum = 100;
% step = int32(size(testcodes, 1) / ptnum);
% 
base_pr = [];
learn_pr = [];
% whrank_pr = zeros(ptnum, 2);


% collect samples
testsamps = [];
testsampslabels = [];
for i=1:length(testgroups)
    
%     if ~(i==9 || i==7)
%         continue;
%     end
%     if length(testgroups{i}) <= testlimit
%         continue;
%     end
    
    % process current code
%     testlabel = i;
    testsampids = testgroups{i};%randsample( testgroups{i}, 50 );
    testsamps = [testsamps; testcodes(testsampids, :)];
    testsampslabels = [testsampslabels; testlabels(testsampids)];
    
    disp(sprintf('Computed %dth test group.', i));
    
end

ranked_labels = [];

if method == 0
    % base distance ranking
    if strcmp(codename, 'mdsh') == 1
        base_dists = hammingDistEfficientNew(testsamps, testcodes, SHparamNew);
        [base_sorted_dist, base_sorted_idx] = sort(base_dists, 2, 'descend');
    else
        base_dists = weightedHam(testsamps, testcodes, w1', 0);
        [base_sorted_dist, base_sorted_idx] = sort(base_dists, 2);
        ranked_labels = testlabels(base_sorted_idx);
    end

end

if method == 1
    % weighted distance ranking
    learn_dists = weightedHam(testsamps, testcodes, W', 0);
%         learn_dists = svmdist(testsamp, testcodes, svmmodel, 1);
    [learn_sorted_dist, learn_sorted_idx] = sort(learn_dists, 2);
    ranked_labels = testlabels(learn_sorted_idx);
end

if method == 2
    whw = 1 ./ fstd;
    %testsamp_un = testcodes_un(testsampids, :);
    whrank_dists = weightedHam(testsamps, testcodes, whw, 0);
    [~, whrank_sorted_idx] = sort(whrank_dists, 2);
    ranked_labels = testlabels(whrank_sorted_idx);
end

ntest = size(testsamps, 1);

interval = 500;
ap = zeros(1,ntest);
pre = zeros(1,ntest);
pt_num = 1 + floor(size(testcodes,1)/interval);
prr = zeros(1, pt_num*2);
for pi = 1:ntest
    h = double(ranked_labels(pi, :) == testsampslabels(pi));
    ind = find(h > 0);
    pn = length(ind);
%     pre(i) = sum(h(1:range))/range;
%      if pn == 0
%      ap(i) = 0;
%      else
%      tep = 0;
%     for j = 1:pn
%     tep = tep+sum(h( 1:ind(j) ))/ind(j);
%     end
%     ap(i) = tep/pn;
%          end
%     clear ind

    prr = prr + PR_new(h', interval);
    clear h;
end

prr = prr' ./ ntest;
    
if method == 0
    pr = [prr(1:pt_num), prr(pt_num+1:end)];
    base_pr = pr;
    save(basecurvefile, 'pr');
end

if method == 1
    pr = [prr(1:pt_num), prr(pt_num+1:end)];
    learn_pr = pr;
    save(learncurvefile, 'pr');
end

if method == 2
    pr = [prr(1:pt_num), prr(pt_num+1:end)];
    save(whrankcurvefile, 'pr');
end
    
    % compute pr values
%     for k=1:size(testsamp,1)    % every test sample
%         cnt = cnt + 1;
%         
%         for j=1:ptnum    % each top result level
%             
%             topnum = double( (j-1)*step + 1 );
%             
%             if method == 0
%                 % intersection value
%                 base_correct_num = length( find( (testlabels(base_sorted_idx(k, 1:topnum)) == i) > 0) ); 
% %                 base_correct_num = length( intersect( base_sorted_idx(k, 1:topnum), dbids ) );
%                 % precision
%                 base_pr(j, 1) = base_pr(j,1) + double(base_correct_num) / topnum;
%                 % recall
%                 base_pr(j, 2) = base_pr(j,2) + double(base_correct_num) / length(testgroups{testlabel});
%             end
%             
%             if method == 1
%                 learn_correct_num = length( find( (testlabels(learn_sorted_idx(k, 1:topnum)) == i) > 0) ); 
% %                 learn_correct_num = length( intersect( learn_sorted_idx(k, 1:topnum), dbids ) );
%                 learn_pr(j, 1) = learn_pr(j,1) + double(learn_correct_num) / topnum;
%                 learn_pr(j, 2) = learn_pr(j,2) + double(learn_correct_num) / length(testgroups{testlabel});
%             end
%             
%             if method == 2
%                 whrank_correct_num = length( find( (testlabels(whrank_sorted_idx(k, 1:topnum)) == i) > 0) ); 
%                 whrank_pr(j, 1) = whrank_pr(j,1) + double(whrank_correct_num) / topnum;
%                 whrank_pr(j, 2) = whrank_pr(j,2) + double(whrank_correct_num) / length(testgroups{testlabel});
%             end
%             
%         end
%     end
    

    

% base_pr = base_pr ./ cnt;
% learn_pr = learn_pr ./ cnt;
% whrank_pr = whrank_pr ./ cnt;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% backup code

%     if(showres==1)
%         % show ranked results in images, top 5
%         figure('Name', 'Base Results')
%         for k=1:5
%             res = traindata(base_sorted_idx(1,k), :);
%             res = reshape(res, imgsz, imgsz);
%             subplot(1,5,k)
%             imshow(res)
%             hold on
%         end
%     end




