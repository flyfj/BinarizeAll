
% batch evaluate all methods

clear

use_data = 1;
code_type = 1;
codenames = {'sh', 'itq', 'lsh', 'mdsh', 'iso'};
codes = [1 2 3 5];
bits = [16, 32, 48, 96, 128];

dataname = 'mnist';

todraw = 0;

for i=1:length(codes)
    for j=1:length(bits)
        
        codename = codenames{codes(i)};
        [base_pr, learn_pr] = main_processor(dataname, codename, bits(j), 2);
        
        if todraw == 1
            % draw pr curve
            close all
            xlabel('Recall')
            ylabel('Precision')
            hold on
            axis([0 1 0 1]);
            hold on
            grid on
            plot(base_pr(:,2), base_pr(:,1), 'b-')
            hold on
            plot(learn_pr(:,2), learn_pr(:,1), 'r-')
            hold on
            legend('Base', 'Weighted')
            pause
        end
        
    end
end


% regular ml data format, treat as two groups, one for the same class, the
% other for all different ones
%use_data = 3;