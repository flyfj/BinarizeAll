
function [base_pr, learn_pr] = main_processor(dataname, codename, nbits, tolearn)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main entrance for owh
% 1. load data
% 2. compute hash code using different methods
% 3. learn weights
% 4. evaluation


%% load binary codes

addpath(genpath('svm'));

% dataname = 'mnist';
% codename = 'sh';
% nbits = 32;

disp(['Dataset: ' dataname ' Code: ' codename 'bits: ' nbits]);

disp('Loading binary codes...');

code_params.nbits = nbits;
codefile = sprintf('data/%s_codes/%s_%s_%db.mat', dataname, dataname, codename, nbits);
basecurvefile = sprintf('res/%s_%s_%db_pr.mat', dataname, codename, nbits);
learncurvefile = sprintf('res/%s_%s_%db_pr_weighted.mat', dataname, codename, nbits);

load(codefile);

labels = unique(trainlabels);

% split data into train and test: 50-50
traingroups = cell(length(labels), 1);
testgroups = cell(length(labels), 1);
for i=1:length(labels)
    clsids = find(trainlabels == labels(i));
    traingroups{i,1} = clsids;
    clsids = find(testlabels == labels(i));
    testgroups{i,1} = clsids;
end

disp('Binary code loaded.');


%% generate similarity pairs
if tolearn == 1
    disp('Generating training pairs...');

    if ~exist('sim_data', 'var')
        sim_data = genSimData(traingroups, 'triplet', 5000);
        %sim_data2 = genSimData(testgroups, 'triplet', 4000);
        %sim_data = [sim_data; sim_data2];
    end

    disp('Generating training pairs done.');
end

%% learn weights using ranksvm formulation

if tolearn == 1

    disp('Learning weights...');

    % now use relative attribute code

    svm_type = 'ranksvm';

    % construct parameters for svm code
    svm_opt.lin_cg = 0; % not use conjugate gradient
    svm_opt.iter_max_Newton = 200;   % Maximum number of Newton steps
    svm_opt.prec = 0.0000000001;    %   prec: Stopping criterion
    w_0 = zeros(1, code_params.nbits);   % initial weights
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
        code_dist_vecs = double(2*code_dist_vecs - 1);

        % construct ordering and similarity matrix: pair_num X sample_num
        O = zeros(triplet_num, size(code_dist_vecs, 1));
        S = zeros(triplet_num, size(code_dist_vecs, 1));
        % Each row of O should contain exactly one +1 and one -1.
        for i=1:length(sim_idx)

            S(i, sim_idx(i,1)) = -1;
            S(i, sim_idx(i,2)) = 1;
        end

        for i=1:length(ordered_idx)

            O(i, ordered_idx(i,1)) = 1;
            O(i, ordered_idx(i,2)) = -1;
        end

        % use rank-svm first
        C_S = ones(1, triplet_num) * 0.1;
        C_O = ones(1, triplet_num) * 0.1;
        %W = ranksvm(code_dist_vecs, O, C_O', w_0', svm_opt); 

        % online mode

    %     W = w_0';
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

    end

    if strcmp(svm_type, 'normal')

        % use hamming distance vector as sample
        global X;
        sampnum = size(sim_data{1,1}, 1);
        X = zeros(sampnum, code_params.nbits);
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
        svmmodel = svmtrain( double(newlabels(1:trainsz,:)), double(X(1:trainsz,:)), '-t 2');

        %svmoption = statset('Display', 'iter');
        %svmmodel = svmtrain(X, newlabels, 'kernel_function', 'quadratic', 'showplot', 0, 'options', svmoption);
        %[W,b0,obj] = primal_svm(1,newlabels,1,svm_opt)

        % test classification performance

        %pred_labels = svmclassify(svmmodel, X);
        [pred_labels, accuracy, scores] = svmpredict(newlabels(trainsz+1:end,:), X(trainsz+1:end,:), svmmodel);


        % test a query
        testid = 44;
        gtlabels = testgroups{testlabels(testid,1), 1};
        % compute difference vector with each testcode
        test_dist_vecs = repmat(testcodes(testid,:), size(testcodes, 1), 1);
        test_dist_vecs = abs(test_dist_vecs - testcodes) * 2 - 1;
        test_dist_vecs = double(test_dist_vecs);
        truelabels = -ones(length(testlabels), 1);
        truelabels(gtlabels) = 1;
        [pred_labels, accuracy, scores] = svmpredict(truelabels, test_dist_vecs, svmmodel);
    %     corr_num = intersect( find(pred_labels==1), train_groups{trainlabel(testid,1), 1} );
    %     corr_num = length(corr_num) / length(train_groups{trainlabel(testid,1), 1});

        % use scores to simply evaluate ranking performance
        [scores_sorted, score_sorted_idx] = sort(scores, 1, 'descend');
        score_inters = zeros(2, size(testcodes, 1));
        for i=1:size(testcodes, 1)
            % intersection value
            inter_num = length( intersect( score_sorted_idx(1:i, 1), gtlabels ) );
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

ptnum = 100;
step = int32(size(testcodes, 1) / ptnum);

base_pr = zeros(ptnum, 2);
learn_pr = zeros(ptnum, 2);

W = w1;

cnt = 0;    % count number of curves / samples
for i=1:length(testgroups)
    
    % process current code
    testlabel = i;
    testsamp = testcodes(randsample(testgroups{i}, 15), :);
    
    % base distance ranking
    base_dists = weightedHam(testsamp, testcodes, w1', 0);
    [base_sorted_dist, base_sorted_idx] = sort(base_dists, 2);
    
    % weighted distance ranking
    learn_dists = weightedHam(testsamp, testcodes, W', 1);
    [learn_sorted_dist, learn_sorted_idx] = sort(learn_dists, 2);
    
    dbids = testgroups{testlabel};
    
    % compute pr values
    for k=1:2:size(testsamp,1)    % every sample
        cnt = cnt + 1;
        
        for j=1:ptnum    % each top result level
            
            topnum = double( (j-1)*step + 1 );
            % intersection value
            base_correct_num = length( intersect( base_sorted_idx(k, 1:topnum), dbids ) );
            learn_correct_num = length( intersect( learn_sorted_idx(k, 1:topnum), dbids ) );
            % precision
            base_pr(j, 1) = base_pr(j,1) + double(base_correct_num) / topnum;
            learn_pr(j, 1) = learn_pr(j,1) + double(learn_correct_num) / topnum;
            % recall
            base_pr(j, 2) = base_pr(j,2) + double(base_correct_num) / length(dbids);
            learn_pr(j, 2) = learn_pr(j,2) + double(learn_correct_num) / length(dbids);
        end
    end
    
    disp(sprintf('Computed %dth test group.', i));
    
end

base_pr = base_pr ./ cnt;
learn_pr = learn_pr ./ cnt;

% compute average pr
% p_ids = 1:2:size(base_pr,2);
% r_ids = 2:2:size(base_pr,2);
% base_pr = [mean(base_pr(:,p_ids), 2), mean(base_pr(:,r_ids), 2)];
% learn_pr = [mean(learn_pr(:,p_ids), 2), mean(learn_pr(:,r_ids), 2)];

pr = base_pr;
save(basecurvefile, 'pr');

if tolearn == 1
    pr = learn_pr;
    save(learncurvefile, 'pr');
end


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




