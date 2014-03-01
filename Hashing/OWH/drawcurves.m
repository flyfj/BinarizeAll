
%% draw result curves

dataname = 'cifar';

% base code
% lsh
lsh16 = load(sprintf('res/%s_lsh_16b_pr.mat', dataname));
lsh16 = lsh16.pr;
lsh32 = load(sprintf('res/%s_lsh_32b_pr.mat', dataname));
lsh32 = lsh32.pr;
lsh48 = load(sprintf('res/%s_lsh_48b_pr.mat', dataname));
lsh48 = lsh48.pr;
lsh96 = load(sprintf('res/%s_lsh_96b_pr.mat', dataname));
lsh96 = lsh96.pr;
lsh128 = load(sprintf('res/%s_lsh_128b_pr.mat', dataname));
lsh128 = lsh128.pr;

% sh
sh16 = load(sprintf('res/%s_sh_16b_pr.mat', dataname));
sh16 = sh16.pr;
sh32 = load(sprintf('res/%s_sh_32b_pr.mat', dataname));
sh32 = sh32.pr;
sh48 = load(sprintf('res/%s_sh_48b_pr.mat', dataname));
sh48 = sh48.pr;
sh96 = load(sprintf('res/%s_sh_96b_pr.mat', dataname));
sh96 = sh96.pr;
sh128 = load(sprintf('res/%s_sh_128b_pr.mat', dataname));
sh128 = sh128.pr;

% itq
itq16 = load(sprintf('res/%s_itq_16b_pr.mat', dataname));
itq16 = itq16.pr;
itq32 = load(sprintf('res/%s_itq_32b_pr.mat', dataname));
itq32 = itq32.pr;
itq48 = load(sprintf('res/%s_itq_48b_pr.mat', dataname));
itq48 = itq48.pr;
itq96 = load(sprintf('res/%s_itq_96b_pr.mat', dataname));
itq96 = itq96.pr;
itq128 = load(sprintf('res/%s_itq_128b_pr.mat', dataname));
itq128 = itq128.pr;


% mdsh
mdshfile = sprintf('res/%s_mdsh_32b_pr.mat', dataname);


% iso
iso16 = load(sprintf('res/%s_iso_16b_pr.mat', dataname));
iso16 = iso16.pr;
iso32 = load(sprintf('res/%s_iso_32b_pr.mat', dataname));
iso32 = iso32.pr;
iso48 = load(sprintf('res/%s_iso_48b_pr.mat', dataname));
iso48 = iso48.pr;
iso96 = load(sprintf('res/%s_iso_96b_pr.mat', dataname));
iso96 = iso96.pr;
iso128 = load(sprintf('res/%s_iso_128b_pr.mat', dataname));
iso128 = iso128.pr;

% ############################################
% weighted


% ############################################
% draw

sh_d = 0;
lsh_d = 0;
itq_d = 1;
iso_d = 0;

figure('Name', 'PR Curves on MNIST using 32bits')
xlabel('Recall')
ylabel('Precision')
hold on
axis([0,1,0,1]);
hold on
grid on

if sh_d == 1
    plot(sh16(:,2), sh16(:,1), 'g-', 'LineWidth', 2)
    hold on
    plot(sh32(:,2), sh32(:,1), 'r-', 'LineWidth', 2)
    hold on
    plot(sh48(:,2), sh48(:,1), 'k-', 'LineWidth', 2)
    hold on
    plot(sh96(:,2), sh96(:,1), 'c-', 'LineWidth', 2)
    hold on
    plot(sh128(:,2), sh128(:,1), 'm-', 'LineWidth', 2)
    hold on
end

if lsh_d == 1
    % lsh
    plot(lsh16(:,2), lsh16(:,1), 'g-', 'LineWidth', 2)
    hold on
    plot(lsh32(:,2), lsh32(:,1), 'r-', 'LineWidth', 2)
    hold on
    plot(lsh48(:,2), lsh48(:,1), 'k-', 'LineWidth', 2)
    hold on
    plot(lsh96(:,2), lsh96(:,1), 'c-', 'LineWidth', 2)
    hold on
    plot(lsh128(:,2), lsh128(:,1), 'm-', 'LineWidth', 2)
    hold on
end

if itq_d == 1
    % itq
    plot(itq16(:,2), itq16(:,1), 'g-', 'LineWidth', 2)
    hold on
    plot(itq32(:,2), itq32(:,1), 'r-', 'LineWidth', 2)
    hold on
    plot(itq48(:,2), itq48(:,1), 'k-', 'LineWidth', 2)
    hold on
    plot(itq96(:,2), itq96(:,1), 'c-', 'LineWidth', 2)
    hold on
    plot(itq128(:,2), itq128(:,1), 'm-', 'LineWidth', 2)
    hold on
end

if iso_d == 1
    % iso
    plot(iso16(:,2), iso16(:,1), 'g-', 'LineWidth', 2)
    hold on
    plot(iso32(:,2), iso32(:,1), 'r-', 'LineWidth', 2)
    hold on
    plot(iso48(:,2), iso48(:,1), 'k-', 'LineWidth', 2)
    hold on
    plot(iso96(:,2), iso96(:,1), 'c-', 'LineWidth', 2)
    hold on
    plot(iso128(:,2), iso128(:,1), 'm-', 'LineWidth', 2)
    hold on
end

%legend('LSH', 'LSH-Weighted', 'ITQ', 'ITQ-Weighted', 'SH', 'SH-Weighted', 'KSH')
pause


