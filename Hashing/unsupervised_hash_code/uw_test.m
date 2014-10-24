
% get all object folders
root_dir = 'E:\Datasets\RGBD_Dataset\UW\rgbd-obj-dataset\rgbd-dataset\';
subdirs = dir([root_dir '*']);

obj_fns = cell(0,0);
obj_feats = [];
for i=1:length(subdirs)
    cur_dir = subdirs(i);
    if strcmp(cur_dir.name, '.') || strcmp(cur_dir.name, '..')
        continue;
    else
        imgfns = dir([root_dir cur_dir.name '\*crop.png']);
        for j=1:length(imgfns)
            obj_fns = {obj_fns}
        end
        
    end
end