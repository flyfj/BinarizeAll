function num_good = CountGoodConstraints( owh_params, triplets )
%COUNTGOODCONSTRAINTS Summary of this function goes here
%   Detailed explanation goes here
    num_good = 0;
    for i=1:size(triplets,1)
        
        if isempty(triplets{i,1})
            break;
        end
        
        triplet = triplets{i,1};
        
        code_diff_pos = (triplet.query_code - triplet.pos_code).^2;
        code_diff_neg = (triplet.query_code - triplet.neg_code).^2;

        hinge_loss = max( double(code_diff_pos - code_diff_neg) * owh_params.cur_weights' + owh_params.dist_margin, 0);
        if hinge_loss == 0
            num_good = num_good + 1;
        end
    
    end

end

