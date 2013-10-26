
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main entrance for owh
% 1. load data
% 2. compute hash code using different methods
% 3. learn weights
% 4. evaluation

%% init
addpath(genpath('svm'));

%% load raw data

disp('Loading raw feature data...');

% regular ml data format, treat as two groups, one for the same class, the
% other for all different ones
use_data = 3;

[traindata, trainlabels] = loadTrainingData(use_data);
% make label starts from 1
trainlabels = trainlabels + 1;

% use subset as test data
testdata = [];
testlabels = [];

% separate into groups with same labels
unique_label_num = length( unique(trainlabels) );
train_groups = cell(unique_label_num, 1);
for i=1:unique_label_num
    cls_ids = find(trainlabels == i);
    train_ids = cls_ids(1:int32(length(cls_ids)*0.8), 1);
    test_ids = cls_ids(int32(length(cls_ids)*0.8)+1:end, 1);
    % 8-2
    train_groups{i,1} = train_ids;
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

code_type = 1;
traincodes = [];
testcodes = [];

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
   % W = ranksvm(code_dist_vecs, O, C', w_0', svm_opt); 

    W = ranksvm_with_sim(code_dist_vecs, O, S, C_O', C_S', w_0', svm_opt);
    
elseif strcmp(svm_type, 'normal')
    
    % use hamming distance vector as sample
    global X;
    X = zeros(length(sim_data{1,1})+length(sim_data{2,1}), code_params.nbits);
    newlabels = zeros(size(X,1), 1);
    for i=1:length(sim_data{1,1})
        X(i,:) = abs(traincodes(sim_data{1,1}(i,2), :) - traincodes(sim_data{1,1}(i,4), :));
        newlabels(i,1) = 1;
    end
    for i=1:length(sim_data{2,1})
        X(i+length(sim_data{1,1}),:) = abs(traincodes(sim_data{2,1}(i,2), :) - traincodes(sim_data{2,1}(i,4), :));
        newlabels(i+length(sim_data{1,1}), 1) = -1;
    end
    
    svmmodel = svmtrain(double(newlabels), double(X));
    
    %svmoption = statset('Display', 'iter');
    %svmmodel = svmtrain(X, newlabels, 'kernel_function', 'quadratic', 'showplot', 0, 'options', svmoption);
    %[W,b0,obj] = primal_svm(1,newlabels,1,svm_opt)
    
    % test classification performance
    
    %pred_labels = svmclassify(svmmodel, X);
    [pred_labels, accuracy, scores] = svmpredict(newlabels, X, svmmodel);
    
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

% use weights and no weights to compute ranking list for one sample first
% use base code: dist and cls_id
w1 = ones(code_params.nbits, 1);

validConstraintNum(traincodes, w1, sim_data)
validConstraintNum(traincodes, W, sim_data)

% every two columns represent one test sample
numtest = 10;
base_pr = zeros(size(traincodes, 1), 2*numtest);
learn_pr = zeros(size(traincodes, 1), 2*numtest);

for i=1:numtest
    % process current code
    testlabel = testlabels(i+30, :);
    testsamp = testcodes(i+30,:);
    base_dists = weightedHam(testsamp, traincodes, w1');
    [base_sorted_dist, base_sorted_idx] = sort(base_dists, 2);

    % use learned weights
    learn_dists = weightedHam(testsamp, traincodes, W');
    [learn_sorted_dist, learn_sorted_idx] = sort(learn_dists, 2);

    % compute pr values
    for j=1:size(traincodes, 1)
        % intersection value
        base_inter_num = size( intersect( base_sorted_idx(1, 1:j), train_groups{testlabel, 1} ), 1 );
        learn_inter_num = size( intersect( learn_sorted_idx(1, 1:j), train_groups{testlabel, 1} ), 1 );
        % precision
        base_pr(j,2*i-1) = double(base_inter_num) / j;
        learn_pr(j,2*i-1) = double(learn_inter_num) / j;
        % recall
        base_pr(j,2*i) = double(base_inter_num) / size(train_groups{testlabel, 1}, 1);
        learn_pr(j,2*i) = double(learn_inter_num) / size(train_groups{testlabel, 1}, 1);
    end
    
end

% compute average pr
p_ids = 1:2:size(base_pr,2);
r_ids = 2:2:size(base_pr,2);
base_pr = [mean(base_pr(:,p_ids), 2), mean(base_pr(:,r_ids), 2)];
learn_pr = [mean(learn_pr(:,p_ids), 2), mean(learn_pr(:,r_ids), 2)];

% draw precision curve
xlabel('Recall')
ylabel('Precision')
hold on
plot(base_pr(:,2), base_pr(:,1), 'r-')
hold on
plot(learn_pr(:,2), learn_pr(:,1), 'b-')
hold on
legend('base', 'Learned')
pause


