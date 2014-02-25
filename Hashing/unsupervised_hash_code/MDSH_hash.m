
load cifar_split;
[n,d] = size(traindata);
tn = size(testdata,1);
range = 100; 
r = 48;
Sigma = 0.4;


%% MDSH
tic
SHparamNew.nbits = r; % number of bits to code each sample
SHparamNew.sigma=Sigma; % Sigma for the affinity. Different codes for different sigmas!
SHparamNew = trainMDSH(traindata, SHparamNew);
[B,Y] = compressMDSH(traindata, SHparamNew);
time = toc;
[time]
clear B;
Y = sign(Y);


%% test
tic
[tB,tY] = compressMDSH(testdata, SHparamNew);
time = toc;
[time/tn]
clear tB;
tY = sign(tY);

WSim = zeros(n,tn);
for i = 1:tn
    if mod(i,10) == 0
       [i]
    end
    dis = hammingDistEfficientNew(tY(i,:), Y, SHparamNew);
    WSim(:,i) = dis';
end
[temp,order] = sort(WSim,1,'descend');
H = traingnd(order);
clear dis;
clear temp;
clear order;
save mdsh_48  Y tY WSim SHparamNew;


% load mdsh_48;
ap = zeros(1,tn);
pre = zeros(1,tn);
interval = 500;
pt_num = 1+floor(n/interval);
prr = zeros(1,pt_num*2);
for i = 1:tn
    h = double(H(:,i) == testgnd(i));
    ind = find(h > 0);
    pn = length(ind);
    pre(i) = sum(h(1:range))/range;
%     if pn == 0
%         ap(i) = 0;
%     else
%         tep = 0;
%         for j = 1:pn
%             tep = tep+sum(h( 1:ind(j) ))/ind(j);
%         end
%         ap(i) = tep/pn;
%     end
%     clear ind;
   
    %% PR curve
    prr = prr+PR_new(h,interval);
    clear h;
end
prr = prr/tn;
[r, mean(pre,2), mean(ap,2)]

mdsh_prr = prr;
plot(mdsh_prr(pt_num+1:end),mdsh_prr(1:pt_num),'b'); hold on; grid;

