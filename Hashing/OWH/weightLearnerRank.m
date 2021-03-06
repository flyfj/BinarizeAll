function W = weightLearnerRank( W0, traincodes, orderidx )
%WEIGHTLEARNERRANK Summary of this function goes here
%   use ranking formulation to learn weights

% use hinge_loss
C = 0.01;
lambda = 0.1;
eta = 0.1;
W = W0;

objcost = [];

% compute gradient
for i=1:size(orderidx,1)

    diff = (traincodes(orderidx(i,2), :) - traincodes(orderidx(i,1), :))';
    if 1-W'*diff > 0
        continue;
    end
    
    grad = W + C*(1-W'*diff).*(-diff);
    grad = grad - C;
    %coeff = exp(-eta * grad);
    
    W = W - lambda * grad;

end

