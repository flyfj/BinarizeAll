
%% draw result curves

dataname = 'mnist';

% base code
% lsh
lshfile = sprintf('res/%s_lsh_32b_pr.mat', dataname);
% lshpr = load(lshfile);
% lshpr = lshpr.pr;
% sh
shfile = sprintf('res/%s_sh_32b_pr.mat', dataname);
shpr = load(shfile);
shpr = shpr.pr;
% itq
itqfile = sprintf('res/%s_itq_32b_pr.mat', dataname);
itqpr = load(itqfile);
itqpr = itqpr.pr;
% mdsh
mdshfile = sprintf('res/%s_mdsh_32b_pr.mat', dataname);


% ############################################
% weighted


% ############################################
% draw
figure('Name', 'PR Curves on MNIST using 32bits')
xlabel('Recall')
ylabel('Precision')
hold on
axis([0,1,0,1]);
hold on
grid on
plot(shpr(:,2), shpr(:,1), 'g-', 'LineWidth', 2)
hold on
plot(itqpr(:,2), itqpr(:,1), 'b-', 'LineWidth', 2)
hold on
%legend('LSH', 'LSH-Weighted', 'ITQ', 'ITQ-Weighted', 'SH', 'SH-Weighted', 'KSH')
pause


