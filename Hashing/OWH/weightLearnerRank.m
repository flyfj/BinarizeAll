function W = weightLearnerRank( W0, traincodes, orderidx )
%WEIGHTLEARNERRANK Summary of this function goes here
%   use ranking formulation to learn weights

% use hinge_loss
C = 0.1;
lambda = 0.01;
W = W0;

% compute gradient
for i=1:size(orderidx,1)

    diff = (traincodes(orderidx(i,1), :) - traincodes(orderidx(i,2), :))';
    if 1+W'*diff > 0
        continue;
    end
    
    grad = (1+W'*diff).*diff;
    grad = W + grad .* C;
    
    W = W + lambda * grad;

end

