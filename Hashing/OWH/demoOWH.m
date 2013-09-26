% Spectral Hashing
% Y. Weiss, A. Torralba, R. Fergus. 
% Advances in Neural Information Processing Systems, 2008.
%
% CONVENTIONS:
%    data points are row vectors.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% all data vectors are row based

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) Create toy data:
% some parameters
Ntraining = 300; % number training samples
Ntest = 300; % number test samples
averageNumberNeighbors = 100; % number of groundtruth neighbors on training set (average)
aspectratio = 0.5; % aspect ratio between the two axis
loopbits = [2 4 8 16 32]; % try different number of bits for coding

% uniform distribution with larger range
% random high dimension data
feat_dim = 100;
data_range = 100;
Xtraining = rand([Ntraining, feat_dim]) .* data_range;
Xtraining(Ntraining/2:end, :) = Xtraining(Ntraining/2:end, :) + 50;
Xtraining(:,2) = aspectratio * Xtraining(:,2);
Xtest = rand([Ntest, feat_dim]) .* data_range / 2; 
%Xtest(:,2) = aspectratio * Xtest(:,2);

% define ground-truth neighbors (this is only used for the evaluation):
DtrueTraining = distMat(Xtraining);
DtrueTestTraining = distMat(Xtest,Xtraining); % size = [Ntest x Ntraining]
[Dball gt_sorted_idx] = sort(DtrueTraining, 2);
%Dball = mean(Dball(:,averageNumberNeighbors));  % mean distance for neighbors in training set
%WtrueTestTraining = DtrueTestTraining < Dball;  % neighbors for testing sampels in training set
%TrainingNeighbors = DtrueTraining < Dball;

train_pairs = cell(size(Xtraining, 1), 2);
for i=1:size(Xtraining,1)
    % neighbors
    train_pairs{i,1} = gt_sorted_idx(i, 1:averageNumberNeighbors);
    % non-neighbors
    train_pairs{i,2} = gt_sorted_idx(i, averageNumberNeighbors:end);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare parameters

bit_num = 32;

OWHParams.nbits = bit_num;
OWHParams.prev_prev_weights = ones(1, OWHParams.nbits) / OWHParams.nbits;
OWHParams.prev_weights = ones(1, OWHParams.nbits); % t, current weights
OWHParams.cur_weights = OWHParams.prev_weights; % t+1, latest weights
OWHParams.lambda = 0.2;
OWHParams.eta = 0.01;
OWHParams.dist_margin = 0;

old_params = OWHParams;

% randomly generate functions

LSHCoder.funcs = randn(bit_num, feat_dim);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) demo online weighted hashing

% generate base hash code for each sample
train_codes = compress2Base(Xtraining, LSHCoder, 'LSH');
test_codes = compress2Base(Xtest, LSHCoder, 'LSH');

% learn weights in online fasion

% only test for 1st training sample
triplets = cell(averageNumberNeighbors*(Ntraining-averageNumberNeighbors), 1);
cnt = 1;
for i=1:2:size(train_pairs{1,1},2)
    for j=1:3:size(train_pairs{1,2},2)
        triplets{cnt,1}.query_code = train_codes(1, :);
        triplets{cnt,1}.query_id = 1;
        triplets{cnt,1}.pos_code = train_codes(train_pairs{1,1}(1,i), :);
        triplets{cnt,1}.pos_id = train_pairs{1,1}(1,i);
        triplets{cnt,1}.neg_code = train_codes(train_pairs{1,2}(1,j), :);
        triplets{cnt,1}.neg_id = train_pairs{1,2}(1,j);
        cnt = cnt + 1;
    end
end

triplets = triplets(1:cnt-1, :);

before_constraints = CountGoodConstraints(OWHParams, triplets);

OWHParams = weightLearnerAll(OWHParams, triplets);

% for i=1:cnt-1
%     
%     % randomly pick a triplet (training sample, positive sample, negative sample)
%     train_id = max(1, int32( rand(1) * size(train_codes, 1) ));
%     pos_id = max(1, int32( rand(1) * size(train_pairs{train_id, 1}, 2) ));
%     pos_id = int32( train_pairs{train_id, 1}(1, pos_id) );
%     neg_id = max(1, int32( rand(1) * size(train_pairs{train_id, 2}, 2) ));
%     neg_id = int32( train_pairs{train_id, 2}(1, neg_id) );
%     
%     triplet.query_code = train_codes(train_id, :);
%     triplet.pos_code = train_codes(pos_id, :);
%     triplet.neg_code = train_codes(neg_id, :);
%     
%     % update weight
%     OWHParams = weightLearner(OWHParams, triplets{i,1});
%     
%     %disp(OWHParams.cur_weights);
%     disp(['Finish ' num2str(i) 'th update.']);
%     
% end

disp(OWHParams.cur_weights);

after_constraints = CountGoodConstraints(OWHParams, triplets);

% evaluation
% check best neighbors with weighted hamming distance and ground truth
% neighbors

% compute lsh hamming distance
lsh_dist = weightedHam(train_codes, train_codes, ones(1, bit_num));
[lsh_sorted_dist, lsh_sorted_idx] = sort(lsh_dist, 2);
lsh_inters = zeros(2, Ntraining);
for i=1:Ntraining
    % precision
    lsh_inters(1,i) = size( intersect( lsh_sorted_idx(1, 1:i), train_pairs{1,1}(1, :) ), 2 ) / i;
    % recall
    lsh_inters(2,i) = size( intersect( lsh_sorted_idx(1, 1:i), train_pairs{1,1}(1, :) ), 2 ) / averageNumberNeighbors;
end


% compute weighted hamming distance
owh_dist = weightedHam(train_codes, train_codes, OWHParams.cur_weights);
[owh_sorted_dist, owh_sorted_idx] = sort(owh_dist, 2);
owh_inters = zeros(2, Ntraining);
for i=1:Ntraining
    owh_inters(1,i) = size( intersect( owh_sorted_idx(1, 1:i), train_pairs{1,1}(1, :) ), 2 ) / i;
    owh_inters(2,i) = size( intersect( owh_sorted_idx(1, 1:i), train_pairs{1,1}(1, :) ), 2 ) / averageNumberNeighbors;
end

% draw precision curve
xlabel('Recall')
ylabel('Precision')
hold on
plot(lsh_inters(2,:), lsh_inters(1,:), 'r-')
hold on
plot(owh_inters(2,:), owh_inters(1,:), 'b-')
hold on
legend('lsh', 'owh')
pause

