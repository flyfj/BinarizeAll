

%% draw curves for different hash methods

% minst

usedata = 3;

%[ksh_base_pr, ksh_learn_pr] = main_processor(usedata, 4);
[sh_base_pr, sh_learn_pr] = main_processor(usedata, 1);
%[itq_base_pr, itq_learn_pr] = main_processor(usedata, 2);
%[lsh_base_pr, lsh_learn_pr] = main_processor(usedata, 3);


neg_step = 0.07;
ksh_pr = [0.59 0; 0.4 0.04; 0.38 0.05; 0.36 0.1; 0.35 0.15; 0.33 0.18; 0.32 0.2; ...
           0.29 0.3; 0.28 0.35; 0.27 0.4; 0.25 0.45; 0.23 0.5; 0.21 0.55; 0.2 0.6; ...
           0.16 0.7; 0.15 0.8; 0.13 0.9; 0.1 1];

wh_lsh_pr = [0.68 0.005; 0.6 0.025; 0.52 0.1; 0.45 0.2; 0.4 0.3; 0.33 0.5];
wh_sh_pr = [0.82 0; 0.72 0.05; 0.63 0.1; 0.52 0.2; 0.46 0.3; 0.41 0.4; 0.35 0.5];
wh_itq_pr = [0.9 0; 0.78 0.025; 0.75 0.05; 0.7 0.1; 0.64 0.2; 0.59 0.3; 0.51 0.4; 0.48 0.5];

% draw precision curve
figure('Name', 'PR Curves on CIFAR10 using 32bits')
xlabel('Recall')
ylabel('Precision')
hold on
axis([0,1,0,1]);
hold on
grid on
% plot(lsh_base_pr(:,2), lsh_base_pr(:,1), 'r-', 'LineWidth', 2)
% hold on
% plot(lsh_learn_pr(:,2), lsh_learn_pr(:,1), 'rd-')
% hold on
% plot(wh_lsh_pr(:,2), wh_lsh_pr(:,1), 'rs-')
% hold on

% plot(itq_base_pr(:,2), itq_base_pr(:,1), 'b-', 'LineWidth', 2)
% hold on
% plot(itq_learn_pr(:,2), itq_learn_pr(:,1), 'bd-')
% hold on
% plot(wh_itq_pr(:,2), wh_itq_pr(:,1), 'bs-')
% hold on

plot(sh_base_pr(:,2), sh_base_pr(:,1), 'g-', 'LineWidth', 2)
hold on
plot(sh_learn_pr(:,2), sh_learn_pr(:,1), 'gd-')
hold on
% plot(wh_sh_pr(:,2), wh_sh_pr(:,1), 'gs-')
% hold on

% plot(ksh_base_pr(:,2), ksh_base_pr(:,1), 'c-', 'LineWidth', 2)
% %plot(ksh_learn_pr(:,2), ksh_learn_pr(:,1), 'cd-')
% hold on



legend('LSH', 'LSH-Weighted', 'ITQ', 'ITQ-Weighted', 'SH', 'SH-Weighted', 'KSH')
pause