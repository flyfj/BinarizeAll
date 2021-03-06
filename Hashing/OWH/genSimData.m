function sim_data = genSimData( cls_samp_ids, type, pnum, validcls )
%GENSIMDATA Summary of this function goes here
%   generate similarity data
%   cls_samp_nums: cell of C classes, each containing object ids (in training set) or object number for each class
%   type: could be pair-wise or triplet
%   sim_data: cell; for pair, contains two unit, one for similar pair ids,
%   one for dissimilar pair ids; for triplet, contains one unit, each row
%   contains one triplet

cls_num = size(cls_samp_ids, 1);

if strcmp(type, 'pair')
    
    % pair format: 
    % similiar pair: (samp1_cls_id, samp1_obj_id, samp2_cls_id, samp2_obj_id)
    % dissimilar pair: same
    
    sim_data = cell(2,1);
    
    % sample certain number
    sim_pair_num = pnum;
    sim_data{1,1} = zeros(sim_pair_num, 4);
    
    % similar pairs within each class
    for i=1:sim_pair_num
        % randomly select a class
        samp_cls_id = int32( randsample(cls_num, 1) );
        samp_obj_id = int32( randsample(cls_samp_ids{samp_cls_id, 1}, 1) );
        sim_cls_id = samp_cls_id;
        sim_obj_id = 0;
        while 1
            temp_obj_id = int32( randsample(cls_samp_ids{samp_cls_id, 1}, 1) );
            if temp_obj_id ~= samp_obj_id
                sim_obj_id = temp_obj_id;
                break;
            end
        end
        
        sim_data{1,1}(i,:) = [samp_cls_id samp_obj_id sim_cls_id sim_obj_id];
    end

    % get all possible similar pairs
%     for i=1:size(cls_samp_ids,1)
%         for j=1:length(cls_samp_ids{i,1})
%             for k=j:length(cls_samp_ids{i,1})
%                 
%                 sim_data{1,1} = [sim_data{1,1}; i cls_samp_ids{i}(j) i cls_samp_ids{i}(k)];
%             end
%         end
%     end
    
    % balanced samples
    dis_pair_num = size(sim_data{1,1}, 1);
    sim_data{2,1} = zeros(dis_pair_num, 4);
    
    % dissimilar pairs in different classes
    for i=1:dis_pair_num
        samp_cls_id = int32( randsample(cls_num, 1) );
        samp_obj_id = int32( randsample(cls_samp_ids{samp_cls_id}, 1) );
        % select dissimilar sample from different classes
        dis_cls_id = 0;
        while 1
            temp_cls_id = int32( randsample(cls_num, 1) );
            if temp_cls_id ~= samp_cls_id
                dis_cls_id = temp_cls_id;
                break;
            end
        end
        dis_obj_id = int32( randsample(cls_samp_ids{dis_cls_id}, 1) );
        
        sim_data{2,1}(i,:) = [samp_cls_id samp_obj_id dis_cls_id dis_obj_id];
    end
    
    
end

if strcmp(type, 'triplet')
        
        
        % triplet format: (samp_id, sim_id1, sim_id2, dis_id)
        % randomly select subset from same class as positive, the rest as negative
        triplet_num = pnum;
        % 1-2: query; 3-4: 1st sim; 5-6: 2nd sim; 7-8: dis
        sim_data = zeros(triplet_num, 8);
        
        % find valid class
        
        % cfiar: ok 7,9,2,1
        % minst: 2, 
        for i=1:triplet_num
            % select a sample; now, force to learn for class 1
            samp_cls_id = int32( randsample(validcls, 1) );
            samp_obj_id = int32( randsample(cls_samp_ids{samp_cls_id}, 1) ); %randsample(801:851, 1);  
            
            % select similar sample from same class
            sim_cls_id = samp_cls_id;
            sim_obj_id = 0;
            while 1
                temp_obj_id = int32( randsample(cls_samp_ids{samp_cls_id}, 1) );
                if temp_obj_id ~= samp_obj_id
                    sim_obj_id = temp_obj_id;
                    break;
                end
            end
            
            % select another similar sample from same class
            sim_obj_id2 = 0;
            while 1
                temp_obj_id = int32( randsample(cls_samp_ids{samp_cls_id}, 1) );
                if temp_obj_id ~= samp_obj_id && temp_obj_id ~= sim_obj_id
                    sim_obj_id2 = temp_obj_id;
                    break;
                end
            end
            
            % select dissimilar sample from different classes
            dis_cls_id = 0;
            while 1
                temp_cls_id = int32( randsample(cls_num, 1) );
                if temp_cls_id ~= samp_cls_id
                    dis_cls_id = temp_cls_id;
                    break;
                end
            end
            dis_obj_id = int32( randsample(cls_samp_ids{dis_cls_id}, 1) );
            
            % add to collection
            sim_data(i,:) = [samp_cls_id, samp_obj_id, sim_cls_id, sim_obj_id, sim_cls_id, sim_obj_id2, dis_cls_id, dis_obj_id];
            
        end
        
end


end

