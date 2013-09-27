function owh_params = weightLearner( owh_params, triplet )
%WEIGHTLEARNER Summary of this function goes here
%   update weight given new similar or dissimilar pairs

showCostCurve = 0;

code_diff_pos = (triplet.query_code - triplet.pos_code).^2;
code_diff_neg = (triplet.query_code - triplet.neg_code).^2;

% hinge_loss = max( double(code_diff_pos - code_diff_neg) * owh_params.cur_weights' + owh_params.dist_margin, 0);
% if hinge_loss == 0
%     return;
% end

% do iteration until converge
delta = 0.0000001;

disp(['Start cost: ' num2str(ComputeCost(owh_params, triplet))]);

costs = [];

% set last parameters
owh_params.prev_weights = owh_params.cur_weights;

owh_params.cur_weights = ones(1, owh_params.nbits);

for t=1:1000
    
    old_weights = owh_params.cur_weights;
    
    % compute new gradient using current weights
    grad = (owh_params.cur_weights - owh_params.prev_weights) + owh_params.lambda * (double(code_diff_pos - code_diff_neg));
    % update weights
    owh_params.cur_weights = owh_params.cur_weights .* exp(-owh_params.eta * grad);
    % normalize weights
    %owh_params.cur_weights = owh_params.cur_weights ./ sum(owh_params.cur_weights);

    % check if converge
    weight_diff = norm(owh_params.cur_weights - old_weights, 2);
    if( weight_diff < delta )
        break;
    end
    
    cur_cost = ComputeCost(owh_params, triplet);
    costs = [costs cur_cost];
    %disp(['Cost for iteraton ' num2str(t) ' : ' num2str(cur_cost)]);
    
end

disp(['End cost: ' num2str(ComputeCost(owh_params, triplet))]);


if showCostCurve == 1
    % visualize cost change
    plot(1:size(costs,2), costs, 'r-')
    pause
end

if(sum(isnan(owh_params.cur_weights)) > 0)
    disp('Error');
end

end


function cost = ComputeCost(owh_params, triplet)

    code_diff_pos = (triplet.query_code - triplet.pos_code).^2;
    code_diff_neg = (triplet.query_code - triplet.neg_code).^2;

    loss = double(code_diff_pos - code_diff_neg) * owh_params.cur_weights' + owh_params.dist_margin;
    %hinge_loss = max( double(code_diff_pos - code_diff_neg) * owh_params.cur_weights' + owh_params.dist_margin, 0);
    cost = norm(owh_params.cur_weights - owh_params.prev_weights, 2).^2 / 2 + owh_params.lambda * loss;

end
