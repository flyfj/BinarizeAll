function num = validConstraintNum( traincodes, W, sim_data )
%VALIDCONSTRAINTNUM Summary of this function goes here
%   Detailed explanation goes here

% compute how many constraints are satisfied using W
num = 0;
for i=1:length(sim_data{1,1})
    
    if W'*abs(traincodes(sim_data{1,1}(i,2), :) - traincodes(sim_data{1,1}(i,4), :))' < W'*abs(traincodes(sim_data{1,1}(i,2), :) - traincodes(sim_data{1,1}(i,8), :))' && ...
        W'*abs(traincodes(sim_data{1,1}(i,4), :) - traincodes(sim_data{1,1}(i,6), :))' < W'*abs(traincodes(sim_data{1,1}(i,4), :) - traincodes(sim_data{1,1}(i,8), :))' && ...
        W'*abs(traincodes(sim_data{1,1}(i,2), :) - traincodes(sim_data{1,1}(i,6), :))' < W'*abs(traincodes(sim_data{1,1}(i,2), :) - traincodes(sim_data{1,1}(i,8), :))'
        num = num + 1;
    end

end

