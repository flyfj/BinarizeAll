function num = validConstraintNum( traincodes, W, sim_data )
%VALIDCONSTRAINTNUM Summary of this function goes here
%   Detailed explanation goes here

% compute how many constraints are satisfied using W
num = 0;
for i=1:length(sim_data{2,1})
    
    if W'*abs(traincodes(sim_data{2,1}(i,2), :) - traincodes(sim_data{2,1}(i,4), :))' > 0
        num = num + 1;
    end

end

