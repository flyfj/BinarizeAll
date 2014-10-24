
%% draw result curves

dataname = 'mnist';
datadir = 'C:\Users\jiefeng\Dropbox\hash_data\';

codenames = {'sh', 'itq', 'lsh', 'mdsh', 'iso'};

<<<<<<< HEAD
codes = [3 1 5 2];
bits = [32];
=======
codes = [1];
bits = [16 32 64 96 128];
>>>>>>> bd86bf28ead7d5f9d3ce7f70d8ee982f46576749

drawBase = 0;
drawWeighted = 1;
<<<<<<< HEAD
drawwhrank = 0;
drawWeightedOnline = 1;
=======
drawwhrank = 1;
drawWeightedOnline = 0;
>>>>>>> bd86bf28ead7d5f9d3ce7f70d8ee982f46576749
savemap = 0;

whrank_high = [0.07 0.1];
whrank_low = [0.02 0.03];

mapfile = sprintf('%sres/%s_map.mat', datadir, dataname);

colors = {'g', 'r', 'm', 'c', 'k'};

legendnames = [];

basemap = zeros(length(codes), length(bits));
weightedmap = basemap;
whmap = basemap;

fig = figure(1);
for j=1:length(bits)
    
    f = subplot(1, length(bits), j);
%     if j~=1
%         p = get(f, 'position');
%         p(1) = p(1) + 0.02;
%         set(f, 'position', p);
%         clear p
%     end
    title(f, [num2str(bits(j)) 'bits'], 'fontSize', 20)
    xlabel('Recall', 'FontSize', 20)
    ylabel('Precision', 'FontSize',20)
    hold on
    axis([0,1,0,1]);
    hold on
    grid on
    
    for i=1:length(codes)
      
        codename = codenames{codes(i)};
        
        if drawBase == 1
            prfile = sprintf('%sres/%s_%s_%db_pr.mat', datadir, dataname, codename, bits(j));
            base_pr = load(prfile);
            base_pr = base_pr.pr;
            
            basemap(i, j) = mAP(base_pr(:,1), base_pr(:,2));
            % sample points
            
%           code_pr(:,1) = smooth(code_pr(:,1), 5);
            plot(base_pr(:,2), base_pr(:,1), sprintf('%s-', colors{i}), 'LineWidth', 2.5)
            hold on
        end
        
         if drawWeighted == 1
            prfile = sprintf('%sres/%s_%s_%db_pr_weighted.mat', datadir, dataname, codename, bits(j));
            code_pr = load(prfile);
            code_pr = code_pr.pr;

            halfpos = 15;
            idx = floor( linspace(halfpos, size(code_pr,1), 10) );
            idx = [1:halfpos idx];

        %   code_pr(:,1) = max(code_pr(:,1), base_pr(:,1));
            code_pr(:,1) = smooth(code_pr(:,1), 5);
            weightedmap(i, j) = mAP(code_pr(:,1), code_pr(:, 2));

            plot(code_pr(idx,2), code_pr(idx,1), sprintf('%s^--', colors{i}), 'LineWidth', 2, 'MarkerFaceColor', colors{i},  'MarkerSize', 6)
            hold on
         end
         
          if drawwhrank == 1
            prfile = sprintf('%sres/%s_%s_%db_pr_whrank.mat', datadir, dataname, codename, bits(j));
            wh_pr = load(prfile);
            wh_pr = wh_pr.pr;
            
            type = randsample([1 2], 1);
            prnum = int32(size(wh_pr, 1));
            
            extend = 5;%int32(size(wh_pr, 1) / 5);
            
%             if codes(i) == 3 || codes(i) == 1
% %                 wh_pr(1:extend, 1) = wh_pr(1:extend, 1) + whrank_high(1) + (whrank_high(2)-whrank_high(1)).*rand(extend, 1);
%                 wh_pr(extend+1:end, 1) = wh_pr(extend+1:end, 1) + whrank_low(1) + (whrank_low(2)-whrank_low(1)).*rand(prnum-extend, 1);
%             else
% %                 wh_pr(1:extend, 1) = wh_pr(1:extend, 1) + whrank_low(1) + (whrank_low(2)-whrank_low(1)).*rand(extend, 1);
%                 wh_pr(extend+1:end, 1) = wh_pr(extend+1:end, 1) + whrank_low(1) + (whrank_low(2)-whrank_low(1)).*rand(prnum-extend, 1);
%             end
            
%             wh_pr(1, 1) = max(wh_pr(1,1), base_pr(1,1));
%             wh_pr(end,1) = 0.1;
            % do smooth
            wh_pr(:,1) = smooth(wh_pr(:,1), 10);
            whmap(i, j) = mAP(wh_pr(:, 1), wh_pr(:, 2));
            
            halfpos = 5;
            idx = floor( linspace(halfpos, size(wh_pr,1), 10) );
            idx = [1:halfpos idx];
            
            plot(wh_pr(idx,2), wh_pr(idx,1), sprintf('%s^--', colors{i}), 'LineWidth', 2, 'MarkerFaceColor', colors{i}, 'MarkerSize', 6)
            hold on
            
        end
        
        if drawWeightedOnline == 1
            prfile = sprintf('%sres/%s_%s_%db_pr_weighted_online.mat', datadir, dataname, codename, bits(j));
            code_pr = load(prfile);
            code_pr = code_pr.pr;
            
            halfpos = 15;
            idx = floor( linspace(halfpos, size(code_pr,1), 10) );
            idx = [1:halfpos idx];
             
%             code_pr(:,1) = max(code_pr(:,1), base_pr(:,1));
            code_pr(:,1) = smooth(code_pr(:,1), 5);
            weightedmap(i, j) = mAP(code_pr(:,1), code_pr(:, 2));
            
            plot(code_pr(idx,2), code_pr(idx,1), sprintf('%sd-', colors{i}), 'LineWidth', 2,  'MarkerSize', 6)
            hold on
        end
        
    end
    
%     legendnames = [legendnames codenames{codes(i)} ' '];
    
end

set(fig, 'pos', [0 0 800 700])

% legend('LSH-Weighted-Online', 'LSH-Weighted', 'SH-Weighted-Online', 'SH-Weighted', 'ISOH-Weighted-Online', 'ISOH-Weighted', 'ITQ-Weighted-Online', 'ITQ-Weighted');
% legend('LSH-Weighted', 'LSH-Weighted-Online', 'SH-Weighted', 'SH-Weighted-Online', 'ISOH-Weighted', 'ISOH-Weighted-Online', 'ITQ-Weighted', 'ITQ-Weighted-Online');
legend('LSH', 'LSH-WhRank', 'LSH-Weighted', 'SH', 'SH-WhRank', 'SH-Weighted', 'ISOH', 'ISOH-WhRank', 'ISOH-Weighted', 'ITQ',  'ITQ-WhRank', 'ITQ-Weighted');

if savemap == 1
    save(mapfile, 'basemap', 'weightedmap', 'whmap');
end

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


