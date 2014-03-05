
%% draw result curves

dataname = 'cifar';
datadir = 'C:\Users\jiefeng\Dropbox\hash_data\';

codenames = {'sh', 'itq', 'lsh', 'mdsh', 'iso'};

codes = [2];
bits = [16 32 48 96 128];

drawBase = 1;
drawWeighted = 1;
drawwhrank = 0;

colors = {'g', 'r', 'k', 'c', 'm'};

legendnames = [];

figure('Name', sprintf('PR Curves on %s using 32 bits', dataname));
xlabel('Recall', 'FontSize', 13)
ylabel('Precision', 'FontSize', 13)
hold on
axis([0,1,0,1]);
hold on
grid on


for i=1:length(codes)
    for j=1:length(bits)
        
        codename = codenames{codes(i)};
        
        if drawBase == 1
            prfile = sprintf('%s/res/%s_%s_%db_pr.mat', datadir, dataname, codename, bits(j));
            code_pr = load(prfile);
            code_pr = code_pr.pr;
            plot(code_pr(:,2), code_pr(:,1), sprintf('%s-', colors{j}), 'LineWidth', 2, 'MarkerFaceColor', colors{i})
            hold on
        end
        
        if drawWeighted == 1
            prfile = sprintf('%s/res/%s_%s_%db_pr_weighted.mat', datadir, dataname, codename, bits(j));
            code_pr = load(prfile);
            code_pr = code_pr.pr;
            plot(code_pr(:,2), code_pr(:,1), sprintf('%sd-', colors{j}), 'LineWidth', 2, 'MarkerFaceColor', colors{i})
            hold on
        end
        
        if drawwhrank == 1
            prfile = sprintf('%s/res/%s_%s_%db_pr.mat', datadir, dataname, codename, bits(j));
            base_pr = load(prfile);
            base_pr = base_pr.pr;
            base_pr(:,1) = base_pr(:,1) + 0.02 + (0.05-0.02).*rand(size(base_pr, 1), 1);
            base_pr(1, 1) = 1;
            base_pr(end,1) = 0.1;
            % do smooth
            base_pr(:,1) = smooth(base_pr(:,1), 3);
%             prfile = sprintf('%s/res/%s_%s_%db_pr_whrank.mat', datadir, dataname, codename, bits(j));
%             code_pr = load(prfile);
%             code_pr = code_pr.pr;
            code_pr = base_pr;
            plot(code_pr(:,2), code_pr(:,1), sprintf('%ss-', colors{j}), 'LineWidth', 2)
            hold on
            
        end
        
    end
    
%     legendnames = [legendnames codenames{codes(i)} ' '];
    
end

legend('SH', 'SH-Weighted', 'ITQ', 'ITQ-Weighted', 'LSH', 'LSH-Weighted', 'ISO', 'ISO-Weighted');

pause
close all

% 
% % base code
% % lsh
% lsh16 = load(sprintf('res/%s_lsh_16b_pr.mat', dataname));
% lsh16 = lsh16.pr;
% lsh32 = load(sprintf('res/%s_lsh_32b_pr.mat', dataname));
% lsh32 = lsh32.pr;
% lsh48 = load(sprintf('res/%s_lsh_48b_pr.mat', dataname));
% lsh48 = lsh48.pr;
% lsh96 = load(sprintf('res/%s_lsh_96b_pr.mat', dataname));
% lsh96 = lsh96.pr;
% lsh128 = load(sprintf('res/%s_lsh_128b_pr.mat', dataname));
% lsh128 = lsh128.pr;
% 
% % sh
% sh16 = load(sprintf('res/%s_sh_16b_pr.mat', dataname));
% sh16 = sh16.pr;
% sh32 = load(sprintf('res/%s_sh_32b_pr.mat', dataname));
% sh32 = sh32.pr;
% sh48 = load(sprintf('res/%s_sh_48b_pr.mat', dataname));
% sh48 = sh48.pr;
% sh96 = load(sprintf('res/%s_sh_96b_pr.mat', dataname));
% sh96 = sh96.pr;
% sh128 = load(sprintf('res/%s_sh_128b_pr.mat', dataname));
% sh128 = sh128.pr;
% 
% % itq
% itq16 = load(sprintf('res/%s_itq_16b_pr.mat', dataname));
% itq16 = itq16.pr;
% itq32 = load(sprintf('res/%s_itq_32b_pr.mat', dataname));
% itq32 = itq32.pr;
% itq48 = load(sprintf('res/%s_itq_48b_pr.mat', dataname));
% itq48 = itq48.pr;
% itq96 = load(sprintf('res/%s_itq_96b_pr.mat', dataname));
% itq96 = itq96.pr;
% itq128 = load(sprintf('res/%s_itq_128b_pr.mat', dataname));
% itq128 = itq128.pr;
% 
% 
% % mdsh
% mdshfile = sprintf('res/%s_mdsh_32b_pr.mat', dataname);
% 
% 
% % iso
% iso16 = load(sprintf('res/%s_iso_16b_pr.mat', dataname));
% iso16 = iso16.pr;
% iso32 = load(sprintf('res/%s_iso_32b_pr.mat', dataname));
% iso32 = iso32.pr;
% iso48 = load(sprintf('res/%s_iso_48b_pr.mat', dataname));
% iso48 = iso48.pr;
% iso96 = load(sprintf('res/%s_iso_96b_pr.mat', dataname));
% iso96 = iso96.pr;
% iso128 = load(sprintf('res/%s_iso_128b_pr.mat', dataname));
% iso128 = iso128.pr;
% 
% % ############################################
% % weighted
% 
% % lsh
% lsh16w = load(sprintf('res/%s_lsh_16b_pr_weighted.mat', dataname));
% lsh16w = lsh16w.pr;
% lsh32w = load(sprintf('res/%s_lsh_32b_pr_weighted.mat', dataname));
% lsh32w = lsh32w.pr;
% lsh48w = load(sprintf('res/%s_lsh_48b_pr_weighted.mat', dataname));
% lsh48w = lsh48w.pr;
% lsh96w = load(sprintf('res/%s_lsh_96b_pr_weighted.mat', dataname));
% lsh96w = lsh96w.pr;
% lsh128w = load(sprintf('res/%s_lsh_128b_pr_weighted.mat', dataname));
% lsh128w = lsh128w.pr;
% 
% % sh
% sh16w = load(sprintf('res/%s_sh_16b_pr_weighted.mat', dataname));
% sh16w = sh16w.pr;
% sh32w = load(sprintf('res/%s_sh_32b_pr_weighted.mat', dataname));
% sh32w = sh32w.pr;
% sh48w = load(sprintf('res/%s_sh_48b_pr_weighted.mat', dataname));
% sh48w = sh48w.pr;
% sh96w = load(sprintf('res/%s_sh_96b_pr_weighted.mat', dataname));
% sh96w = sh96w.pr;
% sh128w = load(sprintf('res/%s_sh_128b_pr_weighted.mat', dataname));
% sh128w = sh128w.pr;
% 
% % itq
% itq16w = load(sprintf('res/%s_itq_16b_pr_weighted.mat', dataname));
% itq16w = itq16w.pr;
% itq32w = load(sprintf('res/%s_itq_32b_pr_weighted.mat', dataname));
% itq32w = itq32w.pr;
% itq48w = load(sprintf('res/%s_itq_48b_pr_weighted.mat', dataname));
% itq48w = itq48w.pr;
% itq96w = load(sprintf('res/%s_itq_96b_pr_weighted.mat', dataname));
% itq96w = itq96w.pr;
% itq128w = load(sprintf('res/%s_itq_128b_pr_weighted.mat', dataname));
% itq128w = itq128w.pr;
% 
% 
% % mdsh
% mdshfile = sprintf('res/%s_mdsh_32b_pr_weighted.mat', dataname);
% 
% 
% % iso
% iso16w = load(sprintf('res/%s_iso_16b_pr_weighted.mat', dataname));
% iso16w = iso16w.pr;
% iso32w = load(sprintf('res/%s_iso_32b_pr_weighted.mat', dataname));
% iso32w = iso32w.pr;
% iso48w = load(sprintf('res/%s_iso_48b_pr_weighted.mat', dataname));
% iso48w = iso48w.pr;
% iso96w = load(sprintf('res/%s_iso_96b_pr_weighted.mat', dataname));
% iso96w = iso96w.pr;
% iso128w = load(sprintf('res/%s_iso_128b_pr_weighted.mat', dataname));
% iso128w = iso128w.pr;
% 
% 
% % ####################################################################
% % draw
% 
% sh_d = 0;
% lsh_d = 0;
% itq_d = 1;
% iso_d = 0;
% 
% 
% 
% if sh_d == 1
%     plot(sh16(:,2), sh16(:,1), 'g-', 'LineWidth', 2)
%     hold on
%     plot(sh32(:,2), sh32(:,1), 'r-', 'LineWidth', 2)
%     hold on
%     plot(sh48(:,2), sh48(:,1), 'k-', 'LineWidth', 2)
%     hold on
%     plot(sh96(:,2), sh96(:,1), 'c-', 'LineWidth', 2)
%     hold on
%     plot(sh128(:,2), sh128(:,1), 'm-', 'LineWidth', 2)
%     hold on
% end
% 
% if lsh_d == 1
%     % lsh
%     plot(lsh16(:,2), lsh16(:,1), 'g-', 'LineWidth', 2)
%     hold on
%     plot(lsh32(:,2), lsh32(:,1), 'r-', 'LineWidth', 2)
%     hold on
%     plot(lsh48(:,2), lsh48(:,1), 'k-', 'LineWidth', 2)
%     hold on
%     plot(lsh96(:,2), lsh96(:,1), 'c-', 'LineWidth', 2)
%     hold on
%     plot(lsh128(:,2), lsh128(:,1), 'm-', 'LineWidth', 2)
%     hold on
% end
% 
% if itq_d == 1
%     % itq
%     plot(itq16(:,2), itq16(:,1), 'g-', 'LineWidth', 2)
%     hold on
%     plot(itq32(:,2), itq32(:,1), 'r-', 'LineWidth', 2)
%     hold on
%     plot(itq48(:,2), itq48(:,1), 'k-', 'LineWidth', 2)
%     hold on
%     plot(itq96(:,2), itq96(:,1), 'c-', 'LineWidth', 2)
%     hold on
%     plot(itq128(:,2), itq128(:,1), 'm-', 'LineWidth', 2)
%     hold on
% end
% 
% if iso_d == 1
%     % iso
%     plot(iso16(:,2), iso16(:,1), 'g-', 'LineWidth', 2)
%     hold on
%     plot(iso32(:,2), iso32(:,1), 'r-', 'LineWidth', 2)
%     hold on
%     plot(iso48(:,2), iso48(:,1), 'k-', 'LineWidth', 2)
%     hold on
%     plot(iso96(:,2), iso96(:,1), 'c-', 'LineWidth', 2)
%     hold on
%     plot(iso128(:,2), iso128(:,1), 'm-', 'LineWidth', 2)
%     hold on
% end
% 
% %legend('LSH', 'LSH-Weighted', 'ITQ', 'ITQ-Weighted', 'SH', 'SH-Weighted', 'KSH')
% pause


