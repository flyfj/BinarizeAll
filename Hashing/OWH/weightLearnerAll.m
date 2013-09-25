function owh_params = weightLearnerAll( owh_params, triplets )
%WEIGHTLEARNERALL Summary of this function goes here
%   Detailed explanation goes here

showCostCurve = 1;

% do iteration until converge
delta = 0.0000001;

costs = [];

for t=1:10000
    
    old_weights = owh_params.cur_weights;
    
    % compute new gradient using current weights
    total_diff = zeros(1, owh_params.nbits);
    for i=1:size(triplets,1)
        
        triplet = triplets{i,1};
         
        code_diff_pos = (triplet.query_code - triplet.pos_code).^2;
        code_diff_neg = (triplet.query_code - triplet.neg_code).^2;
        
        hinge_loss = max( double(code_diff_pos - code_diff_neg) * owh_params.cur_weights' + owh_params.dist_margin, 0);
        if hinge_loss == 0
            continue;
        else
            % add to total
            total_diff = total_diff + double(code_diff_pos - code_diff_neg);
        end
        
    end
    
    grad = owh_params.cur_weights + owh_params.lambda * total_diff;
    % update weights
    owh_params.cur_weights = owh_params.cur_weights .* exp(-owh_params.eta * grad);
    % normalize weights
    %owh_params.cur_weights = owh_params.cur_weights ./ sum(owh_params.cur_weights);

    % check if converge
    weight_diff = norm(owh_params.cur_weights - old_weights, 2);
    if( weight_diff < delta )
        break;
    end
    
    cur_cost = ComputeCost(owh_params, triplets);
    costs = [costs cur_cost];
    
    disp(['Iter: ' num2str(t) ' done.']);
    
end

if showCostCurve == 1
    % visualize cost change
    plot(1:size(costs,2), costs, 'r-')
    pause
end

if(sum(isnan(owh_params.cur_weights)) > 0)
    disp('Error');
end

end


function cost = ComputeCost(owh_params, triplets)

    cost = 0;
    
    for i=1:size(triplets, 1)
        triplet = triplets{i,1};
        code_diff_pos = (triplet.query_code - triplet.pos_code).^2;
        code_diff_neg = (triplet.query_code - triplet.neg_code).^2;
    
        hinge_loss = max( double(code_diff_pos - code_diff_neg) * owh_params.cur_weights' + owh_params.dist_margin, 0);
        cost = cost + hinge_loss;
    end
    
    cost = cost + norm(owh_params.cur_weights).^2 / 2 + owh_params.lambda * cost;

end

